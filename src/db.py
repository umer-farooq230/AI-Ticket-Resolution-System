"""
db.py

All SQLite access for the ticket RAG system.

Builds on the existing `tickets` table (from ticket_knowledge_base.db) and
adds, via an idempotent migration in init_db():
  - a `source` column on `tickets` (historical / auto / admin)
  - audit columns on `tickets` recording WHY an auto-sent answer was
    trusted (composite retrieval/confidence scores, supporting-match
    count) -- useful later for a "how is the bot doing" dashboard
  - a `pending_review` table: drafts waiting for an admin to approve/edit,
    carrying the full confidence breakdown (not just one number) so an
    admin reviewing it can see *why* the system wasn't sure.

Every function opens its own short-lived connection -- fine at this scale
and keeps the module thread-safe-ish for simple use / a FastAPI app with
a handful of workers.
"""

import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

# columns added on top of the original CSV-derived `tickets` table
TICKET_AUDIT_COLUMNS = [
    ("source", "TEXT DEFAULT 'historical'"),
    ("resolution_confidence", "REAL"),       # composite_confidence at the time it was resolved
    ("resolution_retrieval_score", "REAL"),  # composite_retrieval_score at the time it was resolved
    ("supporting_match_count", "INTEGER"),
    ("risk_flags", "TEXT"),                  # JSON list, should be empty for auto-sent tickets
]

# full column set for pending_review, expressed the same column-migration
# way as TICKET_AUDIT_COLUMNS so adding a field later is a one-line change
# and never requires a manual ALTER TABLE or a fresh DB.
PENDING_REVIEW_COLUMNS = [
    ("subject", "TEXT"),
    ("body", "TEXT"),
    ("draft_answer", "TEXT"),
    ("llm_confidence", "REAL"),
    ("grounding_score", "REAL"),
    ("composite_confidence", "REAL"),
    ("retrieval_top1_score", "REAL"),
    ("composite_retrieval_score", "REAL"),
    ("supporting_match_count", "INTEGER"),
    ("needs_human_review", "INTEGER"),
    ("risk_flags", "TEXT"),
    ("clarifying_question", "TEXT"),
    ("suggested_type", "TEXT"),
    ("suggested_queue", "TEXT"),
    ("suggested_priority", "TEXT"),
    ("gate_failures", "TEXT"),   # JSON list of human-readable reasons auto-send was blocked
    ("retrieved_ticket_ids", "TEXT"),
    ("user_email", "TEXT"),
    ("status", "TEXT DEFAULT 'pending'"),
    ("final_answer", "TEXT"),
    ("created_at", "TEXT"),
    ("resolved_at", "TEXT"),
]


@contextmanager
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Idempotent migration: add audit columns + create/upgrade pending_review."""
    with get_conn(db_path) as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(tickets)")]
        for name, decl in TICKET_AUDIT_COLUMNS:
            if name not in cols:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {name} {decl}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_review (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
        """)
        pending_cols = [r["name"] for r in conn.execute("PRAGMA table_info(pending_review)")]
        for name, decl in PENDING_REVIEW_COLUMNS:
            if name not in pending_cols:
                conn.execute(f"ALTER TABLE pending_review ADD COLUMN {name} {decl}")

        conn.commit()


# ---------------------------------------------------------------- reads --

def fetch_all_tickets(db_path: str) -> list[dict]:
    """All tickets, for (re)building the Chroma index."""
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT id, subject, body, answer, type, queue, priority, source
            FROM tickets
        """).fetchall()
        return [dict(r) for r in rows]


def get_tickets_by_ids(db_path: str, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    with get_conn(db_path) as conn:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, subject, body, answer, type, queue, priority "
            f"FROM tickets WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return [dict(r) for r in rows]


def get_ticket(db_path: str, ticket_id: int) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None


def delete_ticket(db_path: str, ticket_id: int) -> bool:
    """Returns True if a row was actually deleted."""
    with get_conn(db_path) as conn:
        cur = conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
        conn.commit()
        return cur.rowcount > 0


def list_tickets(db_path: str, source: str | None = None,
                  limit: int = 50, offset: int = 0) -> list[dict]:
    """
    For an admin dashboard: browse resolved tickets, optionally filtered by
    source ('historical' | 'auto' | 'admin') -- i.e. "which were answered
    by the LLM on its own vs. by me".
    """
    with get_conn(db_path) as conn:
        if source:
            rows = conn.execute("""
                SELECT id, subject, answer, type, queue, priority, source,
                       resolution_confidence, resolution_retrieval_score
                FROM tickets WHERE source = ?
                ORDER BY id DESC LIMIT ? OFFSET ?
            """, (source, limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, subject, answer, type, queue, priority, source,
                       resolution_confidence, resolution_retrieval_score
                FROM tickets
                ORDER BY id DESC LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------- writes --

def insert_resolved_ticket(db_path: str, subject: str, body: str, answer: str,
                            source: str, type_: str = "Incident",
                            queue: str = "General Inquiry",
                            priority: str = "medium",
                            resolution_confidence: float | None = None,
                            resolution_retrieval_score: float | None = None,
                            supporting_match_count: int | None = None,
                            risk_flags: list | None = None) -> int:
    """
    Add a newly-answered ticket back into the knowledge base so future
    queries can retrieve it. `source` should be 'auto' or 'admin'.
    Returns the new ticket id.
    """
    with get_conn(db_path) as conn:
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM tickets").fetchone()[0]
        conn.execute("""
            INSERT INTO tickets
                (id, subject, body, answer, type, queue, priority, source,
                 resolution_confidence, resolution_retrieval_score,
                 supporting_match_count, risk_flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            next_id, subject, body, answer, type_, queue, priority, source,
            resolution_confidence, resolution_retrieval_score,
            supporting_match_count, json.dumps(risk_flags or []),
        ))
        conn.commit()
        return next_id


def create_pending_review(db_path: str, subject: str, body: str, draft_answer: str,
                           llm_confidence: float, grounding_score: float | None,
                           composite_confidence: float, retrieval_top1_score: float,
                           composite_retrieval_score: float, supporting_match_count: int,
                           needs_human_review: bool, risk_flags: list,
                           clarifying_question: str | None,
                           suggested_type: str, suggested_queue: str, suggested_priority: str,
                           gate_failures: list[str],
                           retrieved_ticket_ids: list[int], user_email: str = "") -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO pending_review
                (subject, body, draft_answer, llm_confidence, grounding_score,
                 composite_confidence, retrieval_top1_score, composite_retrieval_score,
                 supporting_match_count, needs_human_review, risk_flags,
                 clarifying_question, suggested_type, suggested_queue, suggested_priority,
                 gate_failures, retrieved_ticket_ids, user_email, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            subject, body, draft_answer, llm_confidence, grounding_score,
            composite_confidence, retrieval_top1_score, composite_retrieval_score,
            supporting_match_count, int(needs_human_review), json.dumps(risk_flags or []),
            clarifying_question, suggested_type, suggested_queue, suggested_priority,
            json.dumps(gate_failures or []), json.dumps(retrieved_ticket_ids), user_email,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        return cur.lastrowid


def get_pending_review(db_path: str, pending_id: int) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM pending_review WHERE id = ?", (pending_id,)).fetchone()
        return dict(row) if row else None


def list_pending_reviews(db_path: str, status: str = "pending",
                          limit: int = 50, offset: int = 0) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM pending_review WHERE status = ?
            ORDER BY created_at LIMIT ? OFFSET ?
        """, (status, limit, offset)).fetchall()
        return [dict(r) for r in rows]


def count_pending_reviews(db_path: str, status: str = "pending") -> int:
    with get_conn(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM pending_review WHERE status = ?", (status,)
        ).fetchone()[0]


def resolve_pending_review(db_path: str, pending_id: int, final_answer: str,
                            status: str = "approved") -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            UPDATE pending_review
            SET final_answer = ?, status = ?, resolved_at = ?
            WHERE id = ?
        """, (final_answer, status, datetime.now(timezone.utc).isoformat(), pending_id))
        conn.commit()