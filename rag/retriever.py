"""
Upgraded retrieval module — thread-aware RAG retrieval.

Two retrieval paths run for each incoming email:
  1. Vector similarity search (FAISS) — finds semantically related past emails
  2. Thread history lookup (SQLite) — fetches previous messages in the same thread

Both results feed into the LLM generator for context-rich responses.
"""

from dataclasses import dataclass
from typing import Optional

from config.settings import get_settings
from app.core.logger import get_logger
from rag.embedder import embed_text
from rag.vector_store import get_vector_store

logger = get_logger(__name__)


@dataclass
class RetrievedEmail:
    """A single result from FAISS similarity search."""
    vector_id: int
    score: float           # Cosine similarity (0.0–1.0)
    body_cleaned: str
    category: Optional[str] = None
    sender: Optional[str] = None
    subject: Optional[str] = None


@dataclass
class ThreadMessage:
    """A single message from the conversation thread history."""
    role: str              # 'user' or 'assistant'
    content: str
    created_at: Optional[str] = None


def retrieve_similar_emails(
    query_text: str,
    top_k: Optional[int] = None,
) -> list[RetrievedEmail]:
    """
    Retrieve the most semantically similar emails from the vector store.

    Args:
        query_text:  Cleaned email body to search against.
        top_k:       Number of results to return (defaults to settings.top_k_retrieval).

    Returns:
        List of RetrievedEmail objects sorted by similarity score (highest first).
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval

    logger.info(f"Retrieving top-{k} similar emails | query_len={len(query_text)}")

    query_vector = embed_text(query_text)
    store = get_vector_store()
    raw_results = store.search(query_vector, top_k=k)

    if not raw_results:
        logger.warning("Vector store empty or no similar emails found.")
        return []

    results = [
        RetrievedEmail(
            vector_id=r["vector_id"],
            score=r["score"],
            body_cleaned=r.get("body_cleaned", ""),
            category=r.get("category"),
            sender=r.get("sender"),
            subject=r.get("subject"),
        )
        for r in raw_results
    ]

    scores = [r.score for r in results]
    logger.info(
        f"Retrieved {len(results)} similar emails | "
        f"scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}"
    )
    return results


def retrieve_thread_history(thread_id: str, db) -> list[ThreadMessage]:
    """
    Fetch the conversation history for a given thread from the database.

    Messages are returned in chronological order (oldest first) so the LLM
    can read the conversation naturally.

    Args:
        thread_id:  The thread identifier shared across emails in a conversation.
        db:         SQLAlchemy session (injected by FastAPI dependency).

    Returns:
        List of ThreadMessage objects (up to thread_history_limit).
    """
    from app.models.thread_model import ThreadMessage as ThreadMessageORM

    settings = get_settings()
    limit = settings.thread_history_limit

    rows = (
        db.query(ThreadMessageORM)
        .filter(ThreadMessageORM.thread_id == thread_id)
        .order_by(ThreadMessageORM.created_at.asc())
        .limit(limit)
        .all()
    )

    history = [
        ThreadMessage(
            role=row.role.value if hasattr(row.role, "value") else str(row.role),
            content=row.content,
            created_at=str(row.created_at),
        )
        for row in rows
    ]

    logger.info(f"Thread history for {thread_id!r}: {len(history)} messages retrieved")
    return history


def build_context_string(retrieved: list[RetrievedEmail]) -> str:
    """
    Format retrieved emails into a readable context block for the LLM prompt.

    Args:
        retrieved: List of RetrievedEmail results from retrieve_similar_emails().

    Returns:
        Multi-line string ready to be injected into the LLM prompt as {context}.
    """
    if not retrieved:
        return "No similar past emails found in the knowledge base."

    lines = []
    for i, email in enumerate(retrieved, start=1):
        category_label = f"[{email.category}]" if email.category else "[Unknown]"
        lines.append(
            f"--- Example {i} {category_label} (similarity: {email.score:.2f}) ---\n"
            f"{email.body_cleaned[:500]}"  # Cap length to avoid token bloat
        )

    context = "\n\n".join(lines)
    logger.debug(f"Built context ({len(context)} chars) from {len(retrieved)} emails.")
    return context


def build_thread_history_string(history: list[ThreadMessage]) -> str:
    """
    Format thread history into a human-readable conversation block for the LLM.

    Args:
        history: List of ThreadMessage objects from retrieve_thread_history().

    Returns:
        Multi-line string for the {thread_history} placeholder in the prompt.
    """
    if not history:
        return "No prior conversation history for this thread."

    lines = []
    for msg in history:
        role_label = "Customer" if msg.role == "user" else "Support Agent"
        lines.append(f"[{role_label}]: {msg.content[:600]}")

    return "\n\n".join(lines)


def get_average_retrieval_score(retrieved: list[RetrievedEmail]) -> float:
    """
    Compute the mean similarity score from a retrieval result set.

    Returns 0.5 if the list is empty (neutral fallback — not 0 to avoid
    unfairly penalizing emails with no similar past examples).
    """
    if not retrieved:
        return 0.5
    return sum(r.score for r in retrieved) / len(retrieved)
