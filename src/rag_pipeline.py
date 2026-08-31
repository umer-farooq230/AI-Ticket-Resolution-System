"""
rag_pipeline.py

The core flow:

  1. A ticket comes in (subject + body, optionally the customer's email).
  2. Retrieve the top-k most similar past tickets from Chroma.
  3. Ask the LLM to draft an answer from that context (generate_answer),
     with a rich set of self-reported signals -- not just one confidence
     float (see llm_client.py's docstring for why).
  4. Optionally run a second, independent LLM-as-judge pass
     (verify_grounding) that checks the drafted answer against the source
     context alone.
  5. Combine ALL of these into a decision -- see _decide() below for the
     exact logic. Multiple independent signals have to agree before
     anything auto-sends.
  6. auto_sent  -> answer goes to the customer now, ticket is written back
     into SQLite + Chroma with the LLM's own suggested classification.
     pending_admin -> a pending_review row captures the full confidence
     breakdown so an admin can see exactly why the system wasn't sure,
     then notify_admin() fires. admin_approve() later sends the (possibly
     edited) answer and writes it back the same way, with source="admin".

Why not just one threshold on one number? A single similarity score or a
single "confidence" the model reports about its own answer are each easy
to fool in different ways -- a superficially similar retrieved ticket with
a genuinely wrong resolution, or a model that's fluently overconfident.
Requiring retrieval strength, retrieval *consensus* (more than one
supporting match), the drafting model's self-assessment, an independent
grounding check, AND the absence of any explicit risk flag to all agree
means a bad answer has to slip past several different checks at once
before it reaches a customer.
"""

from dataclasses import dataclass, field
from statistics import mean

from src import db
from src import chroma_store
from src import notifier
from src.llm_client import get_llm_client
from src.embeddings import get_embedding_function  # noqa: F401 (re-exported for callers)


@dataclass
class TicketDecision:
    decision: str            # "auto_sent" | "pending_admin"
    answer: str
    # -- confidence breakdown, useful to show an admin or log for later analysis --
    llm_confidence: float
    grounding_score: float | None
    composite_confidence: float
    retrieval_top1_score: float
    composite_retrieval_score: float
    supporting_match_count: int
    needs_human_review: bool
    risk_flags: list = field(default_factory=list)
    clarifying_question: str | None = None
    suggested_type: str = "Incident"
    suggested_queue: str = "General Inquiry"
    suggested_priority: str = "medium"
    gate_failures: list = field(default_factory=list)   # empty when decision == "auto_sent"
    retrieved_ticket_ids: list = field(default_factory=list)
    new_ticket_id: int | None = None       # set when auto_sent
    pending_review_id: int | None = None   # set when pending_admin


class TicketRAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.db_path = config["database"]["sqlite_path"]
        db.init_db(self.db_path)
        self.collection = chroma_store.get_collection(config)
        self.llm = get_llm_client(config)

    # ------------------------------------------------------------ main --

    def process_ticket(self, subject: str, body: str, user_email: str = "") -> TicketDecision:
        rag_cfg = self.config["rag"]

        # -- 1. retrieve --
        retrieval = chroma_store.query_similar(
            self.collection, subject, body, top_k=rag_cfg["top_k"]
        )
        similarities = retrieval["similarities"]
        top1_score = similarities[0] if similarities else 0.0

        per_item_threshold = rag_cfg["per_item_relevance_threshold"]
        supporting_similarities = [s for s in similarities if s >= per_item_threshold]
        supporting_match_count = len(supporting_similarities)
        mean_supporting = mean(supporting_similarities) if supporting_similarities else 0.0

        composite_retrieval_score = (
            rag_cfg["retrieval_top1_weight"] * top1_score
            + rag_cfg["retrieval_mean_weight"] * mean_supporting
        )

        # -- build LLM context from the top matches (full ticket bodies) --
        max_ctx = rag_cfg["max_context_snippets"]
        top_ids = [int(meta["ticket_id"]) for meta in retrieval["metadatas"][:max_ctx]]
        context_snippets = []
        if top_ids:
            full = {t["id"]: t for t in db.get_tickets_by_ids(self.db_path, top_ids)}
            for meta, sim, tid in zip(retrieval["metadatas"][:max_ctx],
                                       similarities[:max_ctx], top_ids):
                context_snippets.append({
                    "subject": meta.get("subject", ""),
                    "body": full.get(tid, {}).get("body", ""),
                    "answer": meta.get("answer", ""),
                    "similarity": sim,
                })

        # -- 2. draft --
        draft = self.llm.generate_answer(
            subject, body, context_snippets, max_snippet_chars=rag_cfg["max_snippet_chars"],
        )
        answer = draft["answer"]
        llm_confidence = draft["confidence"]

        # -- 3. independent grounding check (optional) --
        grounding_score = None
        if rag_cfg["use_grounding_verifier"] and answer:
            verification = self.llm.verify_grounding(
                answer, context_snippets, max_snippet_chars=rag_cfg["max_snippet_chars"],
            )
            grounding_score = verification["grounding_score"]

        composite_confidence = self._composite_confidence(
            llm_confidence, grounding_score, rag_cfg
        )

        gate_failures = self._decide(
            composite_retrieval_score=composite_retrieval_score,
            composite_confidence=composite_confidence,
            supporting_match_count=supporting_match_count,
            draft=draft,
            rag_cfg=rag_cfg,
        )
        auto_ok = not gate_failures

        if auto_ok:
            new_id = db.insert_resolved_ticket(
                self.db_path, subject=subject, body=body, answer=answer, source="auto",
                type_=draft["suggested_type"], queue=draft["suggested_queue"],
                priority=draft["suggested_priority"],
                resolution_confidence=composite_confidence,
                resolution_retrieval_score=composite_retrieval_score,
                supporting_match_count=supporting_match_count,
                risk_flags=draft["risk_flags"],
            )
            chroma_store.add_single_ticket(
                self.collection, new_id, subject, body, answer,
                type_=draft["suggested_type"], queue=draft["suggested_queue"],
                priority=draft["suggested_priority"], source="auto",
            )
            notifier.send_to_customer(user_email, subject, answer)
            return TicketDecision(
                decision="auto_sent", answer=answer,
                llm_confidence=llm_confidence, grounding_score=grounding_score,
                composite_confidence=composite_confidence, retrieval_top1_score=top1_score,
                composite_retrieval_score=composite_retrieval_score,
                supporting_match_count=supporting_match_count,
                needs_human_review=draft["needs_human_review"], risk_flags=draft["risk_flags"],
                clarifying_question=draft["clarifying_question"],
                suggested_type=draft["suggested_type"], suggested_queue=draft["suggested_queue"],
                suggested_priority=draft["suggested_priority"],
                retrieved_ticket_ids=top_ids, new_ticket_id=new_id,
            )

        pending_id = db.create_pending_review(
            self.db_path, subject=subject, body=body, draft_answer=answer,
            llm_confidence=llm_confidence, grounding_score=grounding_score,
            composite_confidence=composite_confidence, retrieval_top1_score=top1_score,
            composite_retrieval_score=composite_retrieval_score,
            supporting_match_count=supporting_match_count,
            needs_human_review=draft["needs_human_review"], risk_flags=draft["risk_flags"],
            clarifying_question=draft["clarifying_question"],
            suggested_type=draft["suggested_type"], suggested_queue=draft["suggested_queue"],
            suggested_priority=draft["suggested_priority"], gate_failures=gate_failures,
            retrieved_ticket_ids=top_ids, user_email=user_email,
        )
        notifier.notify_admin(
            self.config, pending_id, subject, body, answer,
            composite_confidence, composite_retrieval_score,
        )
        return TicketDecision(
            decision="pending_admin", answer=answer,
            llm_confidence=llm_confidence, grounding_score=grounding_score,
            composite_confidence=composite_confidence, retrieval_top1_score=top1_score,
            composite_retrieval_score=composite_retrieval_score,
            supporting_match_count=supporting_match_count,
            needs_human_review=draft["needs_human_review"], risk_flags=draft["risk_flags"],
            clarifying_question=draft["clarifying_question"],
            suggested_type=draft["suggested_type"], suggested_queue=draft["suggested_queue"],
            suggested_priority=draft["suggested_priority"], gate_failures=gate_failures,
            retrieved_ticket_ids=top_ids, pending_review_id=pending_id,
        )

    # --------------------------------------------------------- scoring --

    @staticmethod
    def _composite_confidence(llm_confidence: float, grounding_score: float | None,
                               rag_cfg: dict) -> float:
        if grounding_score is None:
            return llm_confidence
        return (
            rag_cfg["confidence_llm_weight"] * llm_confidence
            + rag_cfg["confidence_grounding_weight"] * grounding_score
        )

    @staticmethod
    def _decide(composite_retrieval_score: float, composite_confidence: float,
                supporting_match_count: int, draft: dict, rag_cfg: dict) -> list[str]:
        """
        Returns a list of human-readable reasons auto-send was blocked --
        empty list means it's clear to auto-send. All of the following
        must hold:
          - composite retrieval score clears the similarity bar
          - composite confidence clears the confidence bar
          - at least `min_supporting_matches` retrieved tickets were
            individually relevant (not just one lucky top-1 hit)
          - the LLM itself didn't flag needs_human_review
          - no risk category was flagged
          - the LLM didn't ask a clarifying question (i.e. it didn't feel
            it was missing information)

        Returning the specific reasons (rather than just True/False) is
        what makes "why does everything end up in admin review" actually
        answerable -- they're stored on the pending_review row and shown
        on each card in the admin UI instead of being a black box.
        """
        failures = []
        if composite_retrieval_score < rag_cfg["similarity_threshold"]:
            failures.append(
                f"retrieval score {composite_retrieval_score:.2f} below threshold "
                f"{rag_cfg['similarity_threshold']:.2f}"
            )
        if composite_confidence < rag_cfg["confidence_threshold"]:
            failures.append(
                f"confidence {composite_confidence:.2f} below threshold "
                f"{rag_cfg['confidence_threshold']:.2f}"
            )
        if supporting_match_count < rag_cfg["min_supporting_matches"]:
            failures.append(
                f"only {supporting_match_count} supporting match(es), "
                f"need {rag_cfg['min_supporting_matches']}"
            )
        if draft["needs_human_review"]:
            failures.append("LLM set needs_human_review=true")
        if draft["risk_flags"]:
            failures.append(f"risk flags: {', '.join(draft['risk_flags'])}")
        if draft["clarifying_question"]:
            failures.append("LLM asked a clarifying question instead of answering")
        return failures

    # -------------------------------------------------------- admin api --

    def admin_approve(self, pending_id: int, final_answer: str | None = None) -> int:
        """
        Admin approves a draft (optionally with edits). Sends the final
        answer to the customer and folds it back into the knowledge base
        (SQLite + Chroma) with source="admin" so future similar tickets
        can be auto-answered. Uses the LLM's suggested classification
        (subject to the admin overriding it later via a future API).
        Returns the new ticket id in the knowledge base.
        """
        pending = db.get_pending_review(self.db_path, pending_id)
        if pending is None:
            raise ValueError(f"No pending_review row with id={pending_id}")

        answer = final_answer if final_answer is not None else pending["draft_answer"]
        db.resolve_pending_review(self.db_path, pending_id, final_answer=answer, status="approved")

        new_id = db.insert_resolved_ticket(
            self.db_path, subject=pending["subject"], body=pending["body"],
            answer=answer, source="admin",
            type_=pending["suggested_type"] or "Incident",
            queue=pending["suggested_queue"] or "General Inquiry",
            priority=pending["suggested_priority"] or "medium",
            resolution_confidence=pending["composite_confidence"],
            resolution_retrieval_score=pending["composite_retrieval_score"],
            supporting_match_count=pending["supporting_match_count"],
            risk_flags=None,  # cleared: a human has now vouched for this answer
        )
        chroma_store.add_single_ticket(
            self.collection, new_id, pending["subject"], pending["body"], answer,
            type_=pending["suggested_type"] or "Incident",
            queue=pending["suggested_queue"] or "General Inquiry",
            priority=pending["suggested_priority"] or "medium", source="admin",
        )
        notifier.send_to_customer(pending["user_email"], pending["subject"], answer)
        return new_id

    def admin_reject(self, pending_id: int, note: str = "") -> None:
        db.resolve_pending_review(self.db_path, pending_id, final_answer=note, status="rejected")

    def delete_ticket(self, ticket_id: int) -> bool:
        """
        Remove a resolved ticket from both SQLite and the Chroma index, so
        it can no longer be retrieved as context for future answers.
        Returns False if no ticket with that id existed.
        """
        deleted = db.delete_ticket(self.db_path, ticket_id)
        if deleted:
            chroma_store.delete_ticket(self.collection, ticket_id)
        return deleted