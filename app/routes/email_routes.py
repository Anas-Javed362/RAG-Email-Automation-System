"""
FastAPI route handlers — async email processing endpoints.

Endpoints:
  POST /email/process  - Run the full RAG pipeline on an incoming email
  POST /email/store    - Store an email in the vector DB (knowledge base seeding)
  GET  /health         - System health check
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.schemas.email_schema import (
    EmailInput,
    EmailStoreInput,
    EmailProcessResponse,
    EmailStoreResponse,
    HealthResponse,
)
from app.services.email_service import process_email, store_email
from app.services.cache_service import get_cache_service
from app.core.logger import get_logger
from config.settings import get_settings
from rag.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/email", tags=["Email Processing"])


@router.post(
    "/process",
    response_model=EmailProcessResponse,
    summary="Process an incoming email",
    description=(
        "Accepts an email payload, runs it through the full RAG pipeline "
        "(clean → classify → retrieve → generate → score), and returns a "
        "categorized response with confidence scoring."
    ),
    status_code=status.HTTP_200_OK,
)
async def process_email_endpoint(
    payload: EmailInput,
    request: Request,
    db: Session = Depends(get_db),
) -> EmailProcessResponse:
    """
    Main endpoint — processes an email through the entire pipeline.

    Accepts an optional `thread_id` for conversation-aware responses.
    When provided, conversation history is retrieved and injected into the
    LLM prompt so responses feel contextual rather than isolated.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        f"POST /email/process | request_id={request_id} | "
        f"sender={payload.sender!r} | thread={payload.thread_id!r}"
    )

    try:
        result = await process_email(payload, db, request_id=request_id)
        return result

    except RuntimeError as exc:
        # LLM exhausted all retries
        logger.error(f"Pipeline failed (RuntimeError): {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {exc}",
        )
    except Exception as exc:
        logger.exception(f"Unexpected error in /email/process: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the email.",
        )


@router.post(
    "/store",
    response_model=EmailStoreResponse,
    summary="Store an email in the vector knowledge base",
    description=(
        "Embeds and stores an email in FAISS without running the full pipeline. "
        "Use this to seed the knowledge base with historical emails."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def store_email_endpoint(
    payload: EmailStoreInput,
    db: Session = Depends(get_db),
) -> EmailStoreResponse:
    """
    Store an email in the vector DB for future retrieval.

    Optionally supply `category` if the correct classification is known.
    """
    logger.info(f"POST /email/store | sender={payload.sender!r} | category={payload.category!r}")

    try:
        result = await store_email(payload, db)
        return result
    except Exception as exc:
        logger.exception(f"Error in /email/store: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store email: {exc}",
        )


# ── Health check lives at the top-level router ─────────────────────────────────
health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns the operational status of all system components.",
)
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Check that the database, vector store, and cache are operational."""
    settings = get_settings()

    # Test DB connectivity
    db_status = "ok"
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"
        logger.error(f"Health check DB error: {exc}")

    # Vector store summary
    vector_store = get_vector_store()
    vector_count = vector_store.total_vectors

    # Cache backend name
    cache = get_cache_service()
    cache_backend = cache.backend_name

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=settings.app_version,
        vector_store=settings.vector_store,
        vector_count=vector_count,
        database=db_status,
        llm_provider=settings.llm_provider,
        cache_backend=cache_backend,
        prompt_version=settings.prompt_version,
    )
