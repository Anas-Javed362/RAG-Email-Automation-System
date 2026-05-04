"""
Email service — the central pipeline orchestrator.

This module wires together all components into a single async pipeline:
  Ingest → Clean → Cache check → Classify → Retrieve (RAG + Thread) →
  Generate → Fuse confidence → Persist → Return response

Design decisions:
  - All I/O-heavy steps (LLM, DB writes) are awaited asynchronously.
  - RequestContext flows through every step for structured logging.
  - Caching is checked before classification to short-circuit identical emails.
  - Thread history and vector retrieval run concurrently (asyncio.gather).
"""

import asyncio
import time
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from config.settings import get_settings
from app.core.logger import get_logger, RequestContext
from app.schemas.email_schema import (
    EmailInput,
    EmailStoreInput,
    EmailProcessResponse,
    EmailStoreResponse,
    ConfidenceBreakdown,
)
from app.models.email_model import Email, EmailCategory
from app.models.thread_model import ThreadMessage, MessageRole
from app.services.cache_service import get_cache_service
from app.services.confidence_service import fuse_confidence
from classifiers.classifier import classify_email
from ingestion.email_cleaner import clean_email_body
from rag.embedder import embed_text
from rag.retriever import (
    retrieve_similar_emails,
    retrieve_thread_history,
    build_context_string,
    build_thread_history_string,
    get_average_retrieval_score,
)
from rag.generator import generate_response
from rag.vector_store import get_vector_store

logger = get_logger(__name__)
settings = get_settings()


async def process_email(
    email_input: EmailInput,
    db: Session,
    request_id: Optional[str] = None,
) -> EmailProcessResponse:
    """
    Process an incoming email through the full RAG pipeline.

    Steps:
        1. Build request context for structured logging
        2. Clean email body
        3. Check response cache (return early if hit)
        4. Classify email (rule-based + LLM)
        5. Retrieve similar emails (FAISS) + thread history (DB) — concurrent
        6. Generate LLM response with retry logic
        7. Compute fused confidence score
        8. Persist email + thread messages to DB
        9. Cache result for future identical requests
        10. Return structured API response

    Args:
        email_input:  Validated inbound email payload.
        db:           SQLAlchemy session from FastAPI dependency.
        request_id:   Trace ID from middleware (auto-generated if None).

    Returns:
        EmailProcessResponse with category, response, confidence, and breakdown.
    """
    req_id = request_id or str(uuid4())
    ctx = RequestContext(
        request_id=req_id,
        sender=email_input.sender,
        subject=email_input.subject or "",
        body_length=len(email_input.body),
        thread_id=email_input.thread_id,
    )

    ctx.log_step(logger, "ingestion", body_len=ctx.body_length, thread_id=ctx.thread_id)

    # ── Step 1: Clean email body ───────────────────────────────────────
    cleaned_body = clean_email_body(email_input.body)
    ctx.log_step(logger, "cleaning", cleaned_len=len(cleaned_body))

    # ── Step 2: Cache check ────────────────────────────────────────────
    cache = get_cache_service()
    cached = cache.get_response(cleaned_body)
    if cached:
        ctx.cache_hit = True
        ctx.log_step(logger, "cache_hit")
        ctx.log_summary(logger)
        return EmailProcessResponse(**cached)

    # ── Step 3: Classify ───────────────────────────────────────────────
    # Run classification in a thread since it may call the LLM synchronously
    classification = await asyncio.get_event_loop().run_in_executor(
        None, lambda: classify_email(cleaned_body, use_llm=True)
    )
    ctx.classification_result = classification.category
    ctx.classification_confidence = classification.confidence
    ctx.log_step(
        logger, "classification",
        category=classification.category,
        confidence=classification.confidence,
        method=classification.method,
    )

    # ── Step 4: Retrieval (concurrent) ─────────────────────────────────
    # Run FAISS similarity search and thread history fetch at the same time
    async def _retrieve_similar():
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: retrieve_similar_emails(cleaned_body)
        )

    async def _retrieve_thread():
        if not email_input.thread_id:
            return []
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: retrieve_thread_history(email_input.thread_id, db)
        )

    similar_emails, thread_history = await asyncio.gather(
        _retrieve_similar(), _retrieve_thread()
    )

    avg_sim = get_average_retrieval_score(similar_emails)
    ctx.similarity_scores = [e.score for e in similar_emails]
    ctx.log_step(
        logger, "retrieval",
        similar_count=len(similar_emails),
        thread_msgs=len(thread_history),
        avg_similarity=round(avg_sim, 3),
    )

    context_str = build_context_string(similar_emails)
    thread_history_str = build_thread_history_string(thread_history)

    # ── Step 5: Generate response ──────────────────────────────────────
    gen_result = await generate_response(
        email_body=cleaned_body,
        sender=email_input.sender,
        subject=email_input.subject or "",
        category=classification.category,
        context=context_str,
        thread_history=thread_history_str,
        prompt_version=settings.prompt_version,
    )
    ctx.llm_latency_ms = gen_result.latency_ms
    ctx.token_usage = gen_result.token_usage
    ctx.log_step(
        logger, "generation",
        latency_ms=round(gen_result.latency_ms, 1),
        llm_self_score=gen_result.llm_self_score,
        tokens=gen_result.token_usage,
    )

    # ── Step 6: Fuse confidence ────────────────────────────────────────
    fused = fuse_confidence(
        classification_confidence=classification.confidence,
        avg_similarity_score=avg_sim,
        llm_self_score=gen_result.llm_self_score,
    )
    ctx.final_confidence = fused.final
    ctx.log_step(
        logger, "confidence_fusion",
        final=fused.final,
        needs_review=fused.needs_review,
    )

    # ── Step 7: Persist to DB ──────────────────────────────────────────
    email_record = Email(
        thread_id=email_input.thread_id,
        sender=email_input.sender,
        subject=email_input.subject,
        body_raw=email_input.body,
        body_cleaned=cleaned_body,
        category=EmailCategory(classification.category),
        classification_confidence=classification.confidence,
        classification_method=classification.method,
        similarity_score=avg_sim,
        response=gen_result.response_text,
        prompt_version=gen_result.prompt_version,
        llm_self_score=gen_result.llm_self_score,
        confidence=fused.final,
        needs_review=fused.needs_review,
    )
    db.add(email_record)
    db.flush()  # Get the email ID before committing

    # ── Step 8: Store thread messages ──────────────────────────────────
    if email_input.thread_id:
        # Store the incoming email as a "user" message
        user_msg = ThreadMessage(
            thread_id=email_input.thread_id,
            role=MessageRole.USER,
            content=cleaned_body,
            email_id=email_record.id,
        )
        # Store the AI response as an "assistant" message
        assistant_msg = ThreadMessage(
            thread_id=email_input.thread_id,
            role=MessageRole.ASSISTANT,
            content=gen_result.response_text,
            email_id=email_record.id,
        )
        db.add_all([user_msg, assistant_msg])

    db.commit()
    ctx.log_step(logger, "db_persist", email_id=email_record.id)

    # ── Step 9: Store vector ───────────────────────────────────────────
    vector_store = get_vector_store()
    vector = await asyncio.get_event_loop().run_in_executor(
        None, lambda: embed_text(cleaned_body)
    )
    vector_id = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: vector_store.add(vector, {
            "body_cleaned": cleaned_body,
            "category": classification.category,
            "sender": email_input.sender,
            "subject": email_input.subject or "",
            "email_id": email_record.id,
        })
    )
    email_record.vector_id = vector_id
    db.commit()

    # ── Step 10: Build response ────────────────────────────────────────
    api_response = EmailProcessResponse(
        request_id=req_id,
        category=classification.category,
        response=gen_result.response_text,
        confidence=fused.final,
        needs_review=fused.needs_review,
        confidence_breakdown=ConfidenceBreakdown(
            classification=fused.classification,
            similarity=fused.similarity,
            llm_self=fused.llm_self,
            final=fused.final,
        ),
        retrieval_count=len(similar_emails),
        thread_id=email_input.thread_id,
        prompt_version=gen_result.prompt_version,
        latency_ms=round(ctx.elapsed_ms, 1),
    )

    # Cache for future identical requests
    cache.set_response(cleaned_body, api_response.model_dump())

    ctx.log_summary(logger)
    return api_response


async def store_email(
    payload: EmailStoreInput,
    db: Session,
) -> EmailStoreResponse:
    """
    Manually store an email in the vector DB without running the full pipeline.

    Used for seeding the knowledge base with historical emails.

    Args:
        payload:  Email data with optional known category.
        db:       SQLAlchemy session.

    Returns:
        EmailStoreResponse with the assigned vector ID.
    """
    cleaned = clean_email_body(payload.body)

    # Embed and store in FAISS
    vector = await asyncio.get_event_loop().run_in_executor(
        None, lambda: embed_text(cleaned)
    )
    vector_store = get_vector_store()
    vector_id = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: vector_store.add(vector, {
            "body_cleaned": cleaned,
            "category": payload.category or "Unknown",
            "sender": payload.sender,
            "subject": payload.subject or "",
        })
    )

    # Persist minimal record to DB
    email_record = Email(
        thread_id=payload.thread_id,
        sender=payload.sender,
        subject=payload.subject,
        body_raw=payload.body,
        body_cleaned=cleaned,
        category=EmailCategory(payload.category) if payload.category else EmailCategory.UNKNOWN,
        confidence=0.0,
        needs_review=False,
        vector_id=vector_id,
    )
    db.add(email_record)
    db.commit()
    db.refresh(email_record)

    logger.info(f"Stored email: vector_id={vector_id} email_id={email_record.id}")

    return EmailStoreResponse(
        message="Email stored successfully in vector DB.",
        vector_id=vector_id,
        email_id=email_record.id,
        thread_id=payload.thread_id,
    )
