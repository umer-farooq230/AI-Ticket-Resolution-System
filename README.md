# Ticket RAG System

```
ticket in -> retrieve similar tickets from Chroma
          -> LLM (Gemini) drafts an answer + a full set of self-reported
             signals (confidence, needs_human_review, risk_flags,
             clarifying_question, suggested type/queue/priority)
          -> a second, independent LLM-as-judge pass grades how well the
             draft is actually grounded in the retrieved context
          -> multi-signal gate (see "Confidence logic" below): only if
             retrieval strength, retrieval consensus, drafting confidence,
             grounding, and the absence of any risk flag ALL agree ->
                 YES -> send answer to customer automatically
                        + save ticket/answer back into SQLite + Chroma
                 NO  -> save a draft in `pending_review` (with the full
                        confidence breakdown), notify admin
                        -> admin edits/approves in the UI -> sent
                        + saved back into SQLite + Chroma
```

A FastAPI backend (`api/`) exposes this over HTTP, and a small static
frontend (`web/`) is served directly from the same app — one process, one
port, no separate frontend build/server.

## Directory layout

```
ticket_rag_system/
├── .env.example              # copy to .env, fill in real secrets
├── requirements.txt
├── config/
│   └── config.yaml            # thresholds, weights, model names, paths (no secrets)
├── data/
│   ├── ticket_knowledge_base.db   # the SQLite KB (28,261 historical tickets)
│   └── chroma_db/                 # created by build_chroma_index.py
├── src/
│   ├── config_loader.py       # loads config.yaml + .env
│   ├── auth.py                 # minimal shared-password admin auth (see below)
│   ├── db.py                  # SQLite: KB reads/writes, pending_review queue, audit columns
│   ├── embeddings.py          # Gemini embeddings + offline mock embedder
│   ├── chroma_store.py        # Chroma collection setup / index / query
│   ├── llm_client.py          # Gemini drafting + grounding-verifier calls + offline mocks
│   ├── notifier.py            # admin notification (console/email) + "send to customer" stub
│   └── rag_pipeline.py        # TicketRAGPipeline: retrieval + scoring + the decision gate
├── api/
│   ├── main.py                 # FastAPI app: ticket + admin endpoints, serves web/ as static files
│   └── schemas.py              # request/response models
├── web/
│   ├── index.html              # user complaint form + admin dashboard (one page, two views)
│   ├── styles.css              # light theme, flat colors
│   └── app.js                  # no framework, no build step -- talks to the API with fetch()
├── scripts/
│   └── build_chroma_index.py   # (re)builds the vector index from the SQLite KB
└── tests/
    └── test_pipeline.py        # end-to-end tester, runs offline by default
```

## Setup

```bash
cd ticket_rag_system
pip install -r requirements.txt
cp .env.example .env
# edit .env: GEMINI_API_KEY, and ideally ADMIN_PASSWORD / AUTH_TOKEN_SECRET
```

Then, with a real API key in `.env` and `app.use_mock_llm: false` in
`config/config.yaml` (the default), build the real vector index — this
embeds all 28,261 tickets via Gemini, so it will make that many API calls
and take a while / cost some quota:

```bash
python scripts/build_chroma_index.py
```

One-time step (re-run only to fully re-embed, e.g. after changing the
embedding model). New tickets resolved by the live pipeline are added to
the index incrementally as they happen.

## Running it

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/** — that's the whole UI. Interactive API
docs are at `/docs`.

- **Submit a Ticket** tab: the complaint form. Anyone can submit; no login
  needed. There's also a "check ticket status" box to look up a resolved
  ticket by its reference number.
- **Admin** tab: password gate first (see below), then a dashboard with
  two views — **Pending review** (every draft waiting on a human, with its
  full confidence breakdown, risk flags, and an editable draft you can
  tweak before approving) and **Resolved tickets** (browse by source —
  `auto` / `admin` / `historical` — i.e. "which were answered by the LLM
  on its own vs. by me").

### Admin login

The default password is **`admin123`** (only if you haven't set
`ADMIN_PASSWORD` in `.env` — set it before using this anywhere but your
own machine). This is a **single shared password**, not a real
user/admin account system — there's no per-admin identity or session
expiry. It exists so the dashboard isn't wide open while the full
login/accounts system described in the spec is built later; `src/auth.py`
is the only file that would need to change to swap in real auth, since
`require_admin` in `api/main.py` is the single integration point every
`/admin/*` route depends on.

## Confidence logic

A single similarity score, or a single number the model reports about its
own answer, are each easy to fool in different ways — a superficially
similar retrieved ticket with the wrong resolution, or a model that's
fluently overconfident about text it just wrote. The gate requires several
independent signals to agree before anything auto-sends
(`rag_pipeline.py::_decide`):

1. **Composite retrieval score** — not just the top-1 similarity. It's a
   weighted blend of the top-1 score and the *mean similarity of every
   retrieved ticket that individually clears `per_item_relevance_threshold`*
   (`retrieval_top1_weight` / `retrieval_mean_weight` in config). One lucky
   top-1 hit surrounded by irrelevant matches scores lower than several
   genuinely similar tickets agreeing.
2. **Supporting-match count** — at least `min_supporting_matches` retrieved
   tickets must individually clear the relevance bar, not just one.
3. **Drafting confidence** — the LLM's own 0–1 self-report from the same
   call that wrote the answer.
4. **Grounding score** — a *second, independent* LLM call
   (`verify_grounding`) that only sees the drafted answer and the source
   context (not the original prompt) and rates how well the answer is
   actually backed by that context, flagging unsupported claims. A model
   grading its own just-written answer in the same call is a weak check;
   a fresh call with a narrower task is a more independent signal.
   `composite_confidence` blends drafting confidence and grounding score
   (`confidence_llm_weight` / `confidence_grounding_weight`).
5. **needs_human_review / risk_flags / clarifying_question** — three
   explicit override signals from the drafting call. The prompt (see
   `RISK_CATEGORIES` in `src/llm_client.py`) instructs the model to flag
   security, legal, refund, health/safety, self-harm, angry-escalation, and
   policy-exception situations regardless of how confident it otherwise is
   — and to ask a clarifying question instead of guessing when the context
   doesn't clearly cover the ticket. Any of these three being non-empty/true
   blocks auto-send outright, no matter what the scores say.

All five have to line up. Every threshold and weight lives in
`config/config.yaml` under `rag:`, with comments explaining each one. The
drafting call's `suggested_type` / `suggested_queue` / `suggested_priority`
also get used to classify newly resolved tickets when they're written
back to the KB, instead of a hardcoded default.

## Testing without an API key

```bash
USE_MOCK_LLM=true python tests/test_pipeline.py
```

Runs the full flow (retrieve → draft → grounding check → auto-send-or-
escalate → admin approve → KB grows) against throwaway copies of the
DB/Chroma store, using deterministic offline stand-ins for the embedder
and both LLM calls — no network, no API key.

To try the UI itself without a key, start the server with the same flag:

```bash
USE_MOCK_LLM=true uvicorn api.main:app --reload --port 8000
```

Once you have a key, drop `USE_MOCK_LLM=true` (or set it to `false`) to
run against real Gemini calls — currently configured for
`gemini-embedding-001` (embeddings) and `gemini-3.6-flash` (generation +
verifier) in `config/config.yaml`. If either model name doesn't exist for
your account, that's the first thing to check against Gemini's current
model list — this project doesn't validate model names itself, it just
passes whatever's in the config through to the API.
