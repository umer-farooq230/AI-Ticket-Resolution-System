"""
schemas.py

Pydantic models for the API. Kept deliberately plain (no auth/user models
yet -- the spec is a future admin/user login system on top of this; these
schemas are the contract that frontend will eventually talk to).
"""

from pydantic import BaseModel, Field


class TicketSubmitRequest(BaseModel):
    subject: str = Field(..., min_length=1, description="Ticket subject line")
    body: str = Field(..., min_length=1, description="Full customer message")
    user_email: str = Field("", description="Customer's email, used to send the final answer")


class TicketSubmitResponse(BaseModel):
    decision: str                      # "auto_sent" | "pending_admin"
    ticket_reference: int              # ticket id (auto_sent) or pending_review id (pending_admin)
    message: str                       # what to show the customer
    answer: str | None = None          # only populated when decision == "auto_sent"


class PendingReviewSummary(BaseModel):
    id: int
    subject: str
    body: str
    draft_answer: str
    composite_confidence: float | None
    composite_retrieval_score: float | None
    supporting_match_count: int | None
    needs_human_review: bool
    risk_flags: list[str]
    clarifying_question: str | None
    suggested_type: str | None
    suggested_queue: str | None
    suggested_priority: str | None
    gate_failures: list[str]
    retrieved_ticket_ids: list[int]
    user_email: str | None
    status: str
    created_at: str | None


class ApproveRequest(BaseModel):
    final_answer: str | None = Field(
        None, description="Edited answer text. Omit to send the LLM's draft as-is."
    )


class ApproveResponse(BaseModel):
    ticket_id: int
    message: str


class RejectRequest(BaseModel):
    note: str = Field("", description="Optional internal note on why this was rejected")


class TicketSummary(BaseModel):
    id: int
    subject: str
    answer: str
    type: str | None
    queue: str | None
    priority: str | None
    source: str
    resolution_confidence: float | None
    resolution_retrieval_score: float | None


class TicketDetail(BaseModel):
    """Full record for the admin 'view' action -- everything in the tickets row."""
    id: int
    subject: str
    body: str
    answer: str
    type: str | None
    queue: str | None
    priority: str | None
    version: str | None
    language: str | None
    source: str
    resolution_confidence: float | None
    resolution_retrieval_score: float | None
    supporting_match_count: int | None
    risk_flags: list[str]


class DeleteResponse(BaseModel):
    message: str


class AdminLoginRequest(BaseModel):
    password: str


class AdminLoginResponse(BaseModel):
    token: str