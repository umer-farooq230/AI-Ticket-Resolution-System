"""
llm_client.py

Two LLM calls make up the "intelligence" of the pipeline:

1. generate_answer() -- the drafting call. Given the new ticket and the
   retrieved similar past tickets, it drafts a reply AND self-reports a
   rich set of signals about that draft: confidence, whether it thinks a
   human should look at this anyway, which risk categories (if any) apply,
   a classification guess (type/queue/priority) for the ticket, and --
   importantly -- a clarifying_question field it can fill in instead of
   guessing when the context doesn't actually cover the situation.

2. verify_grounding() -- a SECOND, independent call that acts as a critic:
   given ONLY the drafted answer and the source context (not the original
   drafting prompt), it rates how well the answer is actually supported by
   that context, and lists any claims it thinks are not backed by it. This
   exists because asking a model to grade its own answer in the same call
   it wrote the answer in is a weak check -- models are systematically
   overconfident about their own immediately-prior output. A fresh call,
   with a narrower task and no memory of "wanting" the answer to be good,
   is a meaningfully more independent signal. rag_pipeline.py blends this
   with the drafting call's self-reported confidence rather than trusting
   either number alone.

LocalLLMClient talks to a local, OpenAI-API-compatible server (Ollama,
vLLM, LM Studio, ...) serving gpt-oss-20b (or whatever you point it at).
MockLLMClient is a deterministic, network-free stand-in with the same
two-call shape, for offline testing.

A note on JSON reliability: smaller/local models are less consistent
about producing strictly valid JSON than large hosted models. If the
response can't be parsed at all, the safe thing to do is treat it as
"insufficient context" and force human review -- but that means a model
that frequently produces slightly-malformed JSON will look like it's
"never answering" even when its actual draft was fine. _parse_json below
tries several increasingly-forgiving extraction strategies before giving
up, specifically to keep that failure mode rare. If tickets are still
landing in admin review for this reason, it'll show up as
`gate_failures` containing "risk_flags" with an `insufficient_context`
flag and `reasoning` reading "could not parse structured output" --
that's a parsing/prompting issue, not a threshold one.
"""

import json
import re

# Kept in sync with config/config.yaml's `queues:` -- also duplicated here
# (not read from the JSON knowledge base) so this module has no import-time
# dependency on the data files.
KNOWN_QUEUES = [
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges",
    "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]
KNOWN_TYPES = ["Incident", "Problem", "Request", "Change"]
KNOWN_PRIORITIES = ["low", "medium", "high"]

RISK_CATEGORIES = {
    "security": "account compromise, unauthorized access, hacking, data breach",
    "legal": "legal threats, lawsuits, regulatory complaints",
    "refund": "refunds, chargebacks, billing disputes involving money owed back",
    "health_safety": "physical health or safety risk",
    "self_harm": "any indication the customer may be a danger to themself",
    "angry_escalation": "extreme anger, threats to leave/churn publicly, abusive language",
    "policy_exception": "the customer is asking for an exception to stated policy",
    "insufficient_context": "the retrieved context doesn't clearly cover this situation",
}

DRAFTING_SYSTEM_INSTRUCTION = f"""You are a customer support assistant drafting a reply to a \
customer ticket. You will be given the new ticket and a set of similar PAST \
tickets with their resolutions, pulled from the company's knowledge base.

Work through this in order:
1. Read the new ticket carefully.
2. Check whether the past tickets actually cover this situation -- not just \
topically similar, but similar enough that their resolution would actually \
be correct advice here.
3. If they do: draft a reply using ONLY information implied by the past \
tickets. Do not invent policies, prices, timelines, or steps that are not \
supported by the context. In this case, needs_human_review should be false \
and confidence should be high (0.7+) -- do not flag a ticket for human \
review just because it is routine; routine, clearly-covered tickets are \
exactly what you are here to handle on your own.
4. If they don't clearly cover it, or you'd be guessing on any material \
point: still draft your best-effort reply, but set a low confidence and, if \
you genuinely need one specific piece of information from the customer to \
proceed, fill in clarifying_question. Otherwise leave clarifying_question null.

Separately, decide needs_human_review: this should be true whenever ANY of \
the following apply, REGARDLESS of how confident you are in the text you \
drafted:
{chr(10).join(f'  - {k}: {v}' for k, v in RISK_CATEGORIES.items())}
List every category that applies in risk_flags (use the exact category keys \
above, e.g. "security", "refund"); use an empty list if none apply. Do NOT \
set needs_human_review to true for ordinary, well-supported tickets just to \
be cautious -- reserve it for the categories above.

Also classify the ticket (best guess) using ONLY these values:
  - suggested_type: one of {KNOWN_TYPES}
  - suggested_queue: one of {KNOWN_QUEUES}
  - suggested_priority: one of {KNOWN_PRIORITIES}

confidence (0.0-1.0) should reflect how safe this draft is to send to the \
customer with NO human review. If the context is a clear, direct match and \
none of the risk categories apply, confidence should be high. If \
needs_human_review is true, confidence should not exceed 0.5.

Respond with ONLY a single JSON object and nothing else -- no markdown code \
fences, no commentary before or after it. Exactly this shape:
{{
  "answer": "<drafted reply text>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>",
  "needs_human_review": <bool>,
  "risk_flags": [<zero or more category keys>],
  "clarifying_question": <string or null>,
  "suggested_type": "<one of the allowed types>",
  "suggested_queue": "<one of the allowed queues>",
  "suggested_priority": "<one of the allowed priorities>"
}}
"""

VERIFIER_SYSTEM_INSTRUCTION = """You are a strict fact-checking reviewer. You will be given \
a DRAFTED ANSWER and the SOURCE CONTEXT it was supposed to be based on \
(nothing else -- you do not have access to the original ticket or any \
other knowledge). Your only job is to judge how well the answer is \
supported by the source context.

Rules:
- Any claim, instruction, policy, price, or promise in the answer that is \
NOT clearly backed by the source context counts against grounding, even if \
it sounds plausible.
- Generic pleasantries ("thank you for reaching out") don't need grounding.
- If the source context is empty or clearly unrelated to the answer, \
grounding_score must be 0.0-0.2.

Respond with ONLY a single JSON object and nothing else -- no markdown code \
fences, no commentary before or after it. Exactly this shape:
{
  "grounding_score": <float 0.0-1.0>,
  "unsupported_claims": [<short strings, empty list if none>]
}
"""


def _build_context_block(context_snippets: list[dict], max_chars: int) -> str:
    parts = []
    for i, s in enumerate(context_snippets, start=1):
        body = (s.get("body") or "")[:max_chars]
        answer = (s.get("answer") or "")[:max_chars]
        parts.append(
            f"--- Past ticket {i} (similarity={s.get('similarity', 0):.2f}) ---\n"
            f"Subject: {s.get('subject', '')}\n"
            f"Customer message: {body}\n"
            f"Resolution given: {answer}\n"
        )
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _json_candidates(text: str):
    """
    Yields progressively more aggressive attempts to recover a JSON object
    from a model response, roughly in order of how much we trust each one.
    Smaller/local models are more prone to wrapping JSON in prose, using
    markdown fences, or leaving a stray trailing comma -- each of those is
    individually easy to recover from, so we try each rather than giving up
    (and forcing a human-review escalation) on the first failure.
    """
    stripped = _strip_code_fences(text)
    yield stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        block = match.group(0)
        yield block
        # trailing commas before a closing bracket/brace are the single most
        # common local-model JSON mistake -- strip them and try again
        yield re.sub(r",\s*([\]}])", r"\1", block)


def _parse_json(text: str, defaults: dict) -> dict:
    for candidate in _json_candidates(text or ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {**defaults, **parsed}
    return {**defaults, "reasoning": "could not parse structured output"}


def _normalize_draft(parsed: dict) -> dict:
    return {
        "answer": str(parsed.get("answer", "")).strip(),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
        "reasoning": str(parsed.get("reasoning", "")),
        "needs_human_review": bool(parsed.get("needs_human_review", False)),
        "risk_flags": [f for f in (parsed.get("risk_flags") or []) if f in RISK_CATEGORIES],
        "clarifying_question": parsed.get("clarifying_question") or None,
        "suggested_type": parsed.get("suggested_type") if parsed.get("suggested_type") in KNOWN_TYPES else "Incident",
        "suggested_queue": parsed.get("suggested_queue") if parsed.get("suggested_queue") in KNOWN_QUEUES else "General Inquiry",
        "suggested_priority": parsed.get("suggested_priority") if parsed.get("suggested_priority") in KNOWN_PRIORITIES else "medium",
    }


def _chat_json(client, model: str, system: str, user: str, temperature: float) -> str:
    """
    Calls chat.completions.create asking for JSON-mode output, but falls
    back to a plain call if the server/model rejects the response_format
    parameter (not every local OpenAI-compatible server supports it). The
    strict "JSON only" instruction in the system prompt plus _parse_json's
    multiple recovery strategies are what actually carry the reliability
    here -- response_format is a nice-to-have on top of that, not a
    dependency.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
        )
    return response.choices[0].message.content or ""


class LocalLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, verifier_model: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._verifier_model = verifier_model or model

    def generate_answer(self, subject: str, body: str, context_snippets: list[dict],
                         max_snippet_chars: int = 800) -> dict:
        context_block = _build_context_block(context_snippets, max_snippet_chars)
        user_prompt = (
            f"NEW TICKET\nSubject: {subject}\nCustomer message: {body}\n\n"
            f"SIMILAR PAST TICKETS\n{context_block if context_block else '(none found)'}"
        )
        text = _chat_json(self._client, self._model, DRAFTING_SYSTEM_INSTRUCTION, user_prompt, 0.2)

        defaults = {"answer": "", "confidence": 0.0, "needs_human_review": True,
                    "risk_flags": ["insufficient_context"], "clarifying_question": None,
                    "suggested_type": "Incident", "suggested_queue": "General Inquiry",
                    "suggested_priority": "medium"}
        return _normalize_draft(_parse_json(text, defaults))

    def verify_grounding(self, answer: str, context_snippets: list[dict],
                          max_snippet_chars: int = 800) -> dict:
        context_block = _build_context_block(context_snippets, max_snippet_chars)
        user_prompt = (
            f"DRAFTED ANSWER\n{answer}\n\n"
            f"SOURCE CONTEXT\n{context_block if context_block else '(none provided)'}"
        )
        text = _chat_json(self._client, self._verifier_model, VERIFIER_SYSTEM_INSTRUCTION, user_prompt, 0.0)

        parsed = _parse_json(text, {"grounding_score": 0.0, "unsupported_claims": []})
        return {
            "grounding_score": max(0.0, min(1.0, float(parsed.get("grounding_score", 0.0) or 0.0))),
            "unsupported_claims": parsed.get("unsupported_claims") or [],
        }


class MockLLMClient:
    """
    Offline stand-in with the same two-call shape as LocalLLMClient.
    - generate_answer(): confidence derived from retrieval similarity;
      risk categories detected via whole-word keyword matching.
    - verify_grounding(): grounding_score derived from whether the answer
      text actually overlaps with the provided context (a crude but
      real check, not a constant).
    """

    RISK_KEYWORD_MAP = {
        "security": ["hack", "hacked", "unauthorized", "breach", "stolen", "compromised"],
        "legal": ["sue", "lawsuit", "lawyer", "legal action"],
        "refund": ["refund", "chargeback", "money back", "reimburse"],
        "health_safety": ["injured", "injury", "unsafe", "fire hazard", "electrocut"],
        "self_harm": ["kill myself", "suicide", "end my life"],
        "angry_escalation": ["unacceptable", "furious", "disgusted", "never again", "cancel my account"],
        "policy_exception": ["make an exception", "waive the fee", "special case"],
    }

    def _detect_risk_flags(self, text: str) -> list[str]:
        flags = []
        for category, keywords in self.RISK_KEYWORD_MAP.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    flags.append(category)
                    break
        return flags

    def generate_answer(self, subject: str, body: str, context_snippets: list[dict],
                         max_snippet_chars: int = 800) -> dict:
        text = f"{subject} {body}".lower()
        risk_flags = self._detect_risk_flags(text)

        if risk_flags:
            return _normalize_draft({
                "answer": "This ticket touches on a sensitive area and has been "
                          "routed to a human agent for review.",
                "confidence": 0.2,
                "reasoning": f"risk categories detected: {risk_flags}",
                "needs_human_review": True,
                "risk_flags": risk_flags,
                "clarifying_question": None,
                "suggested_type": "Incident", "suggested_queue": "General Inquiry",
                "suggested_priority": "high",
            })

        if not context_snippets:
            return _normalize_draft({
                "answer": "Thanks for reaching out -- we don't have a close match "
                          "in our knowledge base yet, so a member of our team will "
                          "follow up shortly.",
                "confidence": 0.0,
                "reasoning": "no similar past tickets found",
                "needs_human_review": True,
                "risk_flags": ["insufficient_context"],
                "clarifying_question": "Could you share a bit more detail about "
                                        "what you were trying to do when this happened?",
                "suggested_type": "Incident", "suggested_queue": "General Inquiry",
                "suggested_priority": "medium",
            })

        best = max(context_snippets, key=lambda s: s.get("similarity", 0))
        sim = best.get("similarity", 0.0)
        return _normalize_draft({
            "answer": best.get("answer", "").strip() or "(no answer text in matched ticket)",
            "confidence": round(sim, 3),
            "reasoning": f"reused answer from most similar past ticket (similarity={sim:.2f})",
            "needs_human_review": sim < 0.5,
            "risk_flags": [],
            "clarifying_question": None,
            "suggested_type": "Incident", "suggested_queue": "General Inquiry",
            "suggested_priority": "medium",
        })

    def verify_grounding(self, answer: str, context_snippets: list[dict],
                          max_snippet_chars: int = 800) -> dict:
        if not context_snippets:
            return {"grounding_score": 0.0, "unsupported_claims": ["no source context provided"]}

        answer_words = set(w for w in re.findall(r"[a-z]+", answer.lower()) if len(w) > 3)
        context_text = " ".join(
            (s.get("answer", "") or "") + " " + (s.get("body", "") or "")
            for s in context_snippets
        ).lower()
        context_words = set(w for w in re.findall(r"[a-z]+", context_text) if len(w) > 3)

        if not answer_words:
            return {"grounding_score": 0.0, "unsupported_claims": ["empty answer"]}

        overlap = len(answer_words & context_words) / len(answer_words)
        unsupported = [] if overlap > 0.5 else ["answer vocabulary diverges from source context"]
        return {"grounding_score": round(overlap, 3), "unsupported_claims": unsupported}


def get_llm_client(config: dict):
    if config["app"]["use_mock_llm"]:
        return MockLLMClient()
    return LocalLLMClient(
        base_url=config["llm"]["base_url"],
        api_key=config["llm"]["api_key"],
        model=config["llm"]["generation_model"],
        verifier_model=config["llm"].get("verifier_model"),
    )