"""
test_pipeline.py

End-to-end smoke test for the whole flow:

    ticket in -> retrieve from Chroma -> LLM drafts + scores confidence
        -> auto-send (if confident)              OR
        -> pending_review + notify admin -> admin_approve() -> sent

Runs in one of two modes:

  1. MOCK MODE (default, no API key / no network needed):
         python tests/test_pipeline.py
     or explicitly:
         USE_MOCK_LLM=true python tests/test_pipeline.py

  2. REAL MODE (uses your local model server, per config.yaml's llm: section):
         USE_MOCK_LLM=false python tests/test_pipeline.py
     (also works if config.yaml has app.use_mock_llm: false -- requires
     your local OpenAI-compatible server, e.g. Ollama, to be running)

It runs against a throwaway copy of the shipped SQLite DB and a throwaway
Chroma directory under a temp folder, so it never touches your real
data/ directory.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src import db
from src import chroma_store
from src.rag_pipeline import TicketRAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_test_config(tmp_dir: Path) -> dict:
    config = load_config()

    # point at throwaway copies so this test never mutates real data/
    src_db = PROJECT_ROOT / "data" / "ticket_knowledge_base.db"
    test_db = tmp_dir / "ticket_knowledge_base.db"
    shutil.copy(src_db, test_db)
    config["database"]["sqlite_path"] = str(test_db)
    config["chroma"]["persist_directory"] = str(tmp_dir / "chroma_db")
    config["chroma"]["collection_name"] = "test_ticket_kb"

    db.init_db(config["database"]["sqlite_path"])  # run the source/pending_review migration
    return config


def seed_chroma_with_sample_tickets(config: dict) -> None:
    """
    Index a small, hand-picked set of tickets (rather than all 28k) so the
    test runs fast and the "known match" scenario below is guaranteed to
    have a close neighbor. Swap this for scripts/build_chroma_index.py's
    full run if you want to test against the entire KB.
    """
    all_tickets = db.fetch_all_tickets(config["database"]["sqlite_path"])
    vpn_related = [t for t in all_tickets if "vpn" in (t["subject"] + t["body"]).lower()][:20]
    billing_related = [t for t in all_tickets if t["queue"] == "Billing and Payments"][:20]
    sample = vpn_related + billing_related
    collection = chroma_store.get_collection(config)
    n = chroma_store.index_tickets(collection, sample)
    print(f"[setup] indexed {n} sample tickets into Chroma for this test run")


def print_decision(label: str, decision) -> None:
    print(f"\n--- {label} ---")
    print(f"decision                  : {decision.decision}")
    print(f"llm_confidence            : {decision.llm_confidence:.3f}")
    print(f"grounding_score           : {decision.grounding_score}")
    print(f"composite_confidence      : {decision.composite_confidence:.3f}")
    print(f"retrieval_top1_score      : {decision.retrieval_top1_score:.3f}")
    print(f"composite_retrieval_score : {decision.composite_retrieval_score:.3f}")
    print(f"supporting_match_count    : {decision.supporting_match_count}")
    print(f"needs_human_review        : {decision.needs_human_review}")
    print(f"risk_flags                : {decision.risk_flags}")
    print(f"clarifying_question       : {decision.clarifying_question}")
    print(f"gate_failures             : {decision.gate_failures}")
    print(f"suggested (type/queue/pri): {decision.suggested_type} / "
          f"{decision.suggested_queue} / {decision.suggested_priority}")
    print(f"retrieved_ticket_ids      : {decision.retrieved_ticket_ids}")
    print(f"answer                    : {decision.answer[:300]}")


def main():
    
    # ignore_cleanup_errors: on Windows, Chroma's sqlite connection inside
    # the temp dir can still be open when Python tries to delete the
    # folder on exit -- don't let that turn into a crash.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        config = make_test_config(tmp_dir)

        mode = "MOCK" if config["app"]["use_mock_llm"] else "REAL (local LLM server)"
        print(f"Running pipeline test in {mode} mode.\n")
        if mode.startswith("REAL"):
            print(f"Targeting {config['llm']['base_url']} -- make sure your local "
                  f"server (e.g. Ollama) is running with "
                  f"{config['llm']['generation_model']} / {config['llm']['embedding_model']} "
                  f"pulled, or this will fail with a connection error.")

        seed_chroma_with_sample_tickets(config)

        pipeline = TicketRAGPipeline(config)

        # ---- Scenario 1: a ticket related to an indexed topic. With the
        #      default (strict) thresholds this should still go to admin --
        #      the mock embedder is a crude bag-of-words stand-in, not a
        #      real semantic model, so treat this as "does retrieval find
        #      *a* related match", not "does it clear the auto-send bar" ----
        decision_1 = pipeline.process_ticket(
            subject="VPN router not connecting",
            body="Our VPN router keeps dropping connection for remote staff. "
                 "We restarted it and reset the firmware but it still won't "
                 "hold a stable connection.",
            user_email="[email protected]",
        )
        print_decision("Scenario 1: known VPN issue", decision_1)
        assert decision_1.retrieval_top1_score > 0, "should retrieve at least some related ticket"

        # ---- Scenario 2: something with no good match in this small index,
        #      should fall back to pending_admin ----
        decision_2 = pipeline.process_ticket(
            subject="My pet hamster chewed through the ethernet cable",
            body="I need a very specific replacement cable and I'm not sure "
                 "which one is compatible with my very unusual setup.",
            user_email="[email protected]",
        )
        print_decision("Scenario 2: novel/no-match ticket", decision_2)

        # ---- Scenario 3: risk keyword should always escalate, regardless
        #      of retrieval score ----
        decision_3 = pipeline.process_ticket(
            subject="Unauthorized access to my account",
            body="Someone hacked my account and I think there's been fraud. "
                 "I will sue if this isn't fixed immediately.",
            user_email="[email protected]",
        )
        print_decision("Scenario 3: security/risk ticket", decision_3)
        assert decision_3.decision == "pending_admin", (
            "Security-flagged tickets must never auto-send"
        )

        # ---- Admin approval flow for whichever tickets ended up pending ----
        for label, decision in [("Scenario 2", decision_2), ("Scenario 3", decision_3)]:
            if decision.decision == "pending_admin":
                print(f"\n--- Admin approving {label} (pending_review id="
                      f"{decision.pending_review_id}) ---")
                edited_answer = decision.answer + "\n\n[Reviewed and approved by admin.]"
                new_ticket_id = pipeline.admin_approve(
                    decision.pending_review_id, final_answer=edited_answer
                )
                print(f"Approved -> written back to KB as ticket id {new_ticket_id}")

        # ---- Verify the KB actually grew ----
        pending_list = db.list_pending_reviews(config["database"]["sqlite_path"], status="approved")
        print(f"\n[verify] {len(pending_list)} pending_review rows now marked 'approved'")
        assert len(pending_list) >= 1

        all_tickets_after = db.fetch_all_tickets(config["database"]["sqlite_path"])
        auto_and_admin = [t for t in all_tickets_after if t["source"] in ("auto", "admin")]
        print(f"[verify] {len(auto_and_admin)} new tickets written back into SQLite "
              f"(source=auto/admin)")
        assert len(auto_and_admin) >= 1

        # ---- Scenario 4: prove the auto_sent branch itself works, using a
        #      relaxed-threshold pipeline against a near-duplicate of an
        #      already-indexed ticket (guaranteed high similarity/consensus) ----
        lenient_config = dict(config)
        lenient_config["rag"] = {
            **config["rag"],
            "similarity_threshold": 0.1,
            "confidence_threshold": 0.1,
            "per_item_relevance_threshold": 0.1,
            "min_supporting_matches": 1,
        }
        lenient_pipeline = TicketRAGPipeline(lenient_config)
        sample_ticket = vpn_sample_for_autosend(config)
        decision_4 = lenient_pipeline.process_ticket(
            subject=sample_ticket["subject"], body=sample_ticket["body"],
            user_email="[email protected]",
        )
        print_decision("Scenario 4: near-duplicate ticket, relaxed thresholds", decision_4)
        assert decision_4.decision == "auto_sent", (
            "a near-duplicate of an indexed ticket with relaxed thresholds should auto-send"
        )
        assert decision_4.new_ticket_id is not None

        print("\nAll checks passed.")


def vpn_sample_for_autosend(config: dict) -> dict:
    tickets = db.fetch_all_tickets(config["database"]["sqlite_path"])
    vpn_related = [t for t in tickets if "vpn" in (t["subject"] + t["body"]).lower()]
    return vpn_related[0]


if __name__ == "__main__":
    main()