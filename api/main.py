"""
main.py

Backend service for the ticket flow, plus the frontend static files.

Run it with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/ for the UI (served from ../web).

Endpoints:
  POST /tickets                        submit a new ticket (the complaint form)
  GET  /tickets/{ticket_id}            look up a resolved ticket by id
  POST /admin/login                    get an admin token (see src/auth.py)
  GET  /admin/tickets                  browse resolved tickets (filter by source)   [admin]
  GET  /admin/tickets/{ticket_id}      full detail on one resolved ticket           [admin]
  DELETE /admin/tickets/{ticket_id}    delete a resolved ticket (SQLite + Chroma)   [admin]
  GET  /admin/pending                  list drafts awaiting admin review            [admin]
  GET  /admin/pending/count            quick badge count for a dashboard            [admin]
  GET  /admin/pending/{pending_id}     full detail on one pending draft             [admin]
  POST /admin/pending/{pending_id}/approve   approve (optionally edited)            [admin]
  POST /admin/pending/{pending_id}/reject    reject with an optional note           [admin]
  GET  /health                         liveness check

[admin] routes require `Authorization: Bearer <token>` from /admin/login.
This is a single shared-password gate, not a real user/admin login system
-- see src/auth.py's docstring.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config_loader import load_config
from src import db, auth
from src.rag_pipeline import TicketRAGPipeline
from api.schemas import (
    TicketSubmitRequest, TicketSubmitResponse,
    PendingReviewSummary, ApproveRequest, ApproveResponse, RejectRequest,
    TicketSummary, TicketDetail, DeleteResponse, AdminLoginRequest, AdminLoginResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config + the pipeline (Chroma connection, LLM client) are built once at
# import time and reused across requests -- both are meant to be long-lived,
# not recreated per-request.
config = load_config()
db_path = config["database"]["sqlite_path"]
pipeline = TicketRAGPipeline(config)

app = FastAPI(title="Ticket RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_admin(authorization: str = Header(default="")) -> None:
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not auth.check_token(token, config):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token")


# ---------------------------------------------------------------- user-facing --
import logging
import traceback
logger = logging.getLogger(__name__)
@app.post("/tickets", response_model=TicketSubmitResponse)
def submit_ticket(req: TicketSubmitRequest):
    try:
        decision = pipeline.process_ticket(
            subject=req.subject, body=req.body, user_email=req.user_email,
        )
        if decision.decision == "auto_sent":
            return TicketSubmitResponse(
                decision="auto_sent",
                ticket_reference=decision.new_ticket_id,
                message="Your ticket has been resolved. Check your email for the answer.",
                answer=decision.answer,
            )
        return TicketSubmitResponse(
            decision="pending_admin",
            ticket_reference=decision.pending_review_id,
            message="Your ticket has been received and is being reviewed by our team. We'll follow up by email.",
            answer=None,
        )
    except Exception as e:
        # Prints the full stack trace to your terminal console
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=TicketSummary)
def get_ticket(ticket_id: int):
    ticket = db.get_ticket(db_path, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketSummary(**ticket)


# -------------------------------------------------------------------- auth --

@app.post("/admin/login", response_model=AdminLoginResponse)
def admin_login(req: AdminLoginRequest):
    token = auth.check_password(req.password, config)
    if not token:
        raise HTTPException(status_code=401, detail="Incorrect password")
    return AdminLoginResponse(token=token)


# --------------------------------------------------------------- admin-facing --

@app.get("/admin/tickets", response_model=list[TicketSummary], dependencies=[Depends(require_admin)])
def list_tickets(source: str | None = None, limit: int = 50, offset: int = 0):
    """
    Browse resolved tickets. `source` filter lets the dashboard show
    "answered automatically by the LLM" (source=auto) separately from
    "answered by me" (source=admin) or the original historical KB.
    """
    if source and source not in ("historical", "auto", "admin"):
        raise HTTPException(status_code=400, detail="source must be historical, auto, or admin")
    rows = db.list_tickets(db_path, source=source, limit=limit, offset=offset)
    return [TicketSummary(**r) for r in rows]


# NOTE: registered before /admin/pending/{pending_id} for the same reason
# /admin/pending/count is -- but /admin/tickets/{ticket_id} has no sibling
# literal route at the same depth, so no ordering hazard here.
@app.get("/admin/tickets/{ticket_id}", response_model=TicketDetail, dependencies=[Depends(require_admin)])
def get_ticket_detail(ticket_id: int):
    """Full record for the admin 'view' action -- includes the full body
    and the confidence/risk audit trail, unlike the public /tickets/{id}."""
    row = db.get_ticket(db_path, ticket_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketDetail(**{**row, "risk_flags": json.loads(row.get("risk_flags") or "[]")})


@app.delete("/admin/tickets/{ticket_id}", response_model=DeleteResponse, dependencies=[Depends(require_admin)])
def delete_ticket(ticket_id: int):
    """
    Deletes a resolved ticket from both SQLite and the Chroma index, so it
    stops showing up as retrieval context for future tickets. Works on
    any source (historical / auto / admin) -- there's no confirmation step
    server-side, that lives in the UI.
    """
    deleted = pipeline.delete_ticket(ticket_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return DeleteResponse(message=f"Ticket {ticket_id} deleted.")


def _pending_row_to_summary(row: dict) -> PendingReviewSummary:
    return PendingReviewSummary(
        id=row["id"], subject=row["subject"], body=row["body"],
        draft_answer=row["draft_answer"],
        composite_confidence=row["composite_confidence"],
        composite_retrieval_score=row["composite_retrieval_score"],
        supporting_match_count=row["supporting_match_count"],
        needs_human_review=bool(row["needs_human_review"]),
        risk_flags=json.loads(row["risk_flags"] or "[]"),
        clarifying_question=row["clarifying_question"],
        suggested_type=row["suggested_type"], suggested_queue=row["suggested_queue"],
        suggested_priority=row["suggested_priority"],
        gate_failures=json.loads(row["gate_failures"] or "[]"),
        retrieved_ticket_ids=json.loads(row["retrieved_ticket_ids"] or "[]"),
        user_email=row["user_email"], status=row["status"], created_at=row["created_at"],
    )


@app.get("/admin/pending", response_model=list[PendingReviewSummary], dependencies=[Depends(require_admin)])
def list_pending(status: str = "pending", limit: int = 50, offset: int = 0):
    """
    What the dashboard polls to see "which tickets does the LLM want me to
    review". Includes the full confidence breakdown so the admin can see
    *why* it wasn't auto-sent (low retrieval consensus? a risk flag? the
    model asking for more info?).
    """
    rows = db.list_pending_reviews(db_path, status=status, limit=limit, offset=offset)
    return [_pending_row_to_summary(r) for r in rows]


# NOTE: this literal route MUST be registered before /admin/pending/{pending_id} --
# FastAPI/Starlette match routes in registration order, so if the parameterized
# route came first, a request for /admin/pending/count would match it with
# pending_id="count" and 422 on int parsing instead of reaching this handler.
@app.get("/admin/pending/count", dependencies=[Depends(require_admin)])
def pending_count(status: str = "pending"):
    return {"status": status, "count": db.count_pending_reviews(db_path, status=status)}


@app.get("/admin/pending/{pending_id}", response_model=PendingReviewSummary, dependencies=[Depends(require_admin)])
def get_pending(pending_id: int):
    row = db.get_pending_review(db_path, pending_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Pending review not found")
    return _pending_row_to_summary(row)


@app.post("/admin/pending/{pending_id}/approve", response_model=ApproveResponse, dependencies=[Depends(require_admin)])
def approve_pending(pending_id: int, req: ApproveRequest):
    """
    Admin approves a draft, optionally with edited text. Sends the final
    answer to the customer and writes it back into the knowledge base.
    """
    try:
        ticket_id = pipeline.admin_approve(pending_id, final_answer=req.final_answer)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApproveResponse(ticket_id=ticket_id, message="Approved and sent to customer.")


@app.post("/admin/pending/{pending_id}/reject", dependencies=[Depends(require_admin)])
def reject_pending(pending_id: int, req: RejectRequest):
    row = db.get_pending_review(db_path, pending_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Pending review not found")
    pipeline.admin_reject(pending_id, note=req.note)
    return {"message": "Rejected."}


# --------------------------------------------------------------------- misc --

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------- frontend --
# Mounted LAST so it acts as a catch-all for anything not matched by the
# explicit API routes above (Starlette checks routes in registration order).
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "web"), html=True), name="web")