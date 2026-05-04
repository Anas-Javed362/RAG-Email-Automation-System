"""
FastAPI application entry point.

Lifecycle:
  startup  → init DB tables, warm up embedding model + vector store
  shutdown → flush logs, close any pending connections

Middleware:
  - RequestIDMiddleware: attaches X-Request-ID to every request/response
  - CORSMiddleware: allows the bundled HTML frontend to call the API locally
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.logger import setup_logging, get_logger
from app.core.middleware import RequestIDMiddleware
from app.database.db import init_db
from app.routes.email_routes import router as email_router, health_router
from config.settings import get_settings

# Start logging before anything else
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # ── Startup ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"RAG Email System v{settings.app_version} starting up...")
    logger.info(f"  LLM provider   : {settings.llm_provider}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  Vector store   : {settings.vector_store}")
    logger.info(f"  Prompt version : {settings.prompt_version}")
    logger.info(f"  Cache backend  : {settings.cache_backend}")
    logger.info(f"  Environment    : {settings.app_env}")
    logger.info("=" * 60)

    # Initialize DB tables
    init_db()

    # Pre-load the embedding model so first request isn't slow
    try:
        from rag.embedder import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model pre-loaded successfully.")
    except Exception as exc:
        logger.warning(f"Could not pre-load embedding model: {exc}")

    # Initialize vector store (loads FAISS index from disk if it exists)
    try:
        from rag.vector_store import get_vector_store
        vs = get_vector_store()
        logger.info(f"Vector store ready: {vs.total_vectors} vectors loaded.")
    except Exception as exc:
        logger.warning(f"Could not initialize vector store: {exc}")

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("RAG Email System shutting down. Goodbye.")


# ── App Instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG-Based Email Automation System",
    description=(
        "Production-grade AI backend that classifies incoming emails and generates "
        "context-aware responses using Retrieval-Augmented Generation (RAG). "
        "Supports thread-aware conversations, multi-signal confidence scoring, "
        "prompt versioning, and caching."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(email_router)
app.include_router(health_router)

# ── Static Files (HTML Frontend) ───────────────────────────────────────────────
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info("Frontend UI mounted at /ui")


# ── Root redirect ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "RAG Email Automation System",
        "version": settings.app_version,
        "docs": "/docs",
        "ui": "/ui/index.html",
        "health": "/health",
    }
