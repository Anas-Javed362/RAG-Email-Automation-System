"""
SQLAlchemy engine and session factory.
Uses SQLite by default, but database_url in settings can point to any SQL engine.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _build_engine():
    """Create SQLAlchemy engine with SQLite-friendly settings."""
    settings = get_settings()
    db_url = settings.database_url

    # SQLite needs special connection args for multi-threaded FastAPI usage
    connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        poolclass=StaticPool if "sqlite" in db_url else None,
        echo=(settings.app_env == "development"),
    )
    logger.info(f"Database engine created: {db_url}")
    return engine


engine = _build_engine()

# Session factory — use as a dependency in FastAPI
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables defined by ORM models."""
    # Import models here to ensure they are registered with Base before create_all
    from app.models import email_model  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")


def get_db():
    """
    FastAPI dependency that yields a DB session and ensures it's closed after use.

    Yields:
        SQLAlchemy Session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
