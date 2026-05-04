"""
Thread history ORM model.

Each row represents one message in a conversation thread.
Used to build conversation-aware LLM prompts.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SAEnum
import enum

from app.database.db import Base


class MessageRole(str, enum.Enum):
    """Speaker role for a thread message."""
    USER = "user"           # Incoming customer email
    ASSISTANT = "assistant" # AI-generated response


class ThreadMessage(Base):
    """
    Stores the ordered history of a conversation thread.

    Columns:
        id          - Auto-incremented primary key
        thread_id   - Groups messages into a single conversation
        role        - 'user' (incoming email) or 'assistant' (AI response)
        content     - The text content of the message
        email_id    - FK reference back to the emails table (informational)
        created_at  - UTC timestamp
    """

    __tablename__ = "thread_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(String(128), nullable=False, index=True)
    role = Column(SAEnum(MessageRole, name="message_role_enum"), nullable=False)
    content = Column(Text, nullable=False)
    email_id = Column(Integer, nullable=True, comment="References emails.id")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<ThreadMessage thread={self.thread_id!r} role={self.role} len={len(self.content)}>"
