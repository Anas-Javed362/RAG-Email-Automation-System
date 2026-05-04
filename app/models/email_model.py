"""
SQLAlchemy ORM model for stored emails — upgraded with thread awareness.

Changes in v2:
  - Added `thread_id` for conversation grouping
  - Added `prompt_version` to track which prompt generated the response
  - Added `llm_self_score` to store the LLM's own confidence rating
  - Added `classification_confidence` and `similarity_score` for audit trail
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy import Enum as SAEnum
import enum

from app.database.db import Base


class EmailCategory(str, enum.Enum):
    """Supported email classification categories."""
    COMPLAINT = "Complaint"
    INQUIRY = "Inquiry"
    SUPPORT = "Support"
    SPAM = "Spam"
    UNKNOWN = "Unknown"


class Email(Base):
    """
    Persists each processed email along with all pipeline outputs.

    Core columns:
        id                      - Auto-incremented primary key
        thread_id               - Groups emails into a conversation thread
        sender                  - Email sender address
        subject                 - Email subject line
        body_raw                - Original body text
        body_cleaned            - Preprocessed body text

    Classification:
        category                - Assigned classification category
        classification_confidence - Classifier's raw confidence before fusion
        classification_method   - Which method was used (rule_based / combined)

    Retrieval:
        similarity_score        - Average cosine similarity from FAISS retrieval

    Generation:
        response                - LLM-generated reply text
        prompt_version          - Prompt template used for generation
        llm_self_score          - LLM's self-rated confidence in its response

    Final scoring:
        confidence              - Fused confidence (0.4*cls + 0.4*sim + 0.2*llm)
        needs_review            - True when confidence < threshold

    Meta:
        vector_id               - Row ID in FAISS index
        created_at              - UTC timestamp of processing
    """

    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(String(128), nullable=True, index=True, comment="Conversation thread identifier")
    sender = Column(String(255), nullable=False, index=True)
    subject = Column(String(512), nullable=True)
    body_raw = Column(Text, nullable=False)
    body_cleaned = Column(Text, nullable=False)

    # Classification outputs
    category = Column(
        SAEnum(EmailCategory, name="email_category_enum"),
        nullable=False,
        default=EmailCategory.UNKNOWN,
    )
    classification_confidence = Column(Float, nullable=True, comment="Raw classifier score before fusion")
    classification_method = Column(String(64), nullable=True)

    # Retrieval signal
    similarity_score = Column(Float, nullable=True, comment="Average FAISS similarity score")

    # Generation outputs
    response = Column(Text, nullable=True)
    prompt_version = Column(String(16), nullable=True, default="v2")
    llm_self_score = Column(Float, nullable=True, comment="LLM's self-rated response confidence")

    # Final fused confidence
    confidence = Column(Float, nullable=False, default=0.0)
    needs_review = Column(Boolean, nullable=False, default=False)

    vector_id = Column(Integer, nullable=True, comment="Row ID in the FAISS index")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<Email id={self.id} thread={self.thread_id!r} sender='{self.sender}' "
            f"category={self.category} confidence={self.confidence:.2f}>"
        )
