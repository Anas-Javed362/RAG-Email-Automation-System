"""
Pydantic schemas — upgraded for production with thread awareness,
request tracing, and detailed confidence breakdowns.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional
from uuid import uuid4


# ── Inbound Request Schemas ────────────────────────────────────────────────────

class EmailInput(BaseModel):
    """
    Schema for incoming email payload.

    New fields in v2:
        thread_id  - Optional conversation thread ID for context-aware generation.
                     The caller should pass the same thread_id for follow-up emails
                     in the same conversation. If omitted, the email is treated as standalone.

    Example:
        {
            "sender": "user@example.com",
            "subject": "Re: Broken product",
            "body": "Still waiting for the replacement...",
            "thread_id": "thread-abc123"
        }
    """
    sender: str = Field(..., description="Email address of the sender")
    subject: Optional[str] = Field(None, description="Email subject line")
    body: str = Field(..., min_length=10, description="Full body of the email")
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID. Supply the same ID for follow-up emails.",
        example="thread-abc123"
    )

    @field_validator("body")
    @classmethod
    def body_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Email body cannot be blank or whitespace only.")
        return value

    @field_validator("sender")
    @classmethod
    def sender_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Sender field cannot be blank.")
        return value


class EmailStoreInput(BaseModel):
    """Payload for the /email/store endpoint."""
    sender: str = Field(..., description="Email address of the sender")
    subject: Optional[str] = Field(None)
    body: str = Field(..., min_length=5)
    category: Optional[str] = Field(None, description="Optional known category for supervised seeding")
    thread_id: Optional[str] = Field(None)


# ── Outbound Response Schemas ──────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    """
    Transparency into how the final confidence score was computed.

    final = 0.4 * classification + 0.4 * similarity + 0.2 * llm_self
    """
    classification: float = Field(..., description="Classifier confidence (rule + LLM)")
    similarity: float = Field(..., description="Average FAISS similarity score")
    llm_self: float = Field(..., description="LLM self-rated confidence in its response")
    final: float = Field(..., description="Fused final score")


class EmailProcessResponse(BaseModel):
    """
    Schema for the response returned after processing an email through the RAG pipeline.

    Fields:
        request_id      - Unique ID for this request (for tracing)
        category        - Predicted classification (Complaint, Inquiry, Support, Spam)
        response        - LLM-generated reply text
        confidence      - Fused confidence score (0.0–1.0)
        needs_review    - True when confidence < configured threshold
        confidence_breakdown - Component-level scores for transparency
        retrieval_count - Number of similar emails retrieved
        thread_id       - Thread ID if this email is part of a conversation
        prompt_version  - Which prompt template was used
        latency_ms      - End-to-end latency in milliseconds
    """
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request trace ID")
    category: str = Field(..., description="Predicted email category")
    response: str = Field(..., description="AI-generated reply for the email")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fused confidence score")
    needs_review: bool = Field(..., description="Whether this email needs human review")
    confidence_breakdown: Optional[ConfidenceBreakdown] = Field(
        None, description="Component scores used to compute final confidence"
    )
    retrieval_count: int = Field(default=0, description="Number of similar emails retrieved")
    thread_id: Optional[str] = Field(None, description="Thread ID if applicable")
    prompt_version: str = Field(default="v2", description="Prompt template version used")
    latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


class EmailStoreResponse(BaseModel):
    """Response returned after storing an email in the vector DB."""
    message: str
    vector_id: int
    email_id: Optional[int] = None
    thread_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    version: str
    vector_store: str
    vector_count: int
    database: str
    llm_provider: str
    cache_backend: str
    prompt_version: str


# ── Stored Email (for GET endpoints / evaluation) ──────────────────────────────

class StoredEmailSchema(BaseModel):
    """Read-only schema for serializing Email ORM objects."""
    id: int
    thread_id: Optional[str]
    sender: str
    subject: Optional[str]
    body_cleaned: str
    category: str
    confidence: float
    classification_confidence: Optional[float]
    similarity_score: Optional[float]
    llm_self_score: Optional[float]
    response: Optional[str]
    needs_review: bool
    prompt_version: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
