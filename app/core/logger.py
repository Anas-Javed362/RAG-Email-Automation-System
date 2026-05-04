"""
Structured request-level logging with request ID, latency, and token tracking.

Upgrades over v1:
  - RequestContext dataclass binds request_id to all log lines
  - log_pipeline_step() emits structured key=value metrics per step
  - log_request_summary() writes a single summary line at the end of each request
"""

import logging
import logging.handlers
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging with console + rotating file handlers.

    Returns:
        Root logger ready for use.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.handlers:
        return root_logger

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file (5 MB, keep 3 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "rag_email.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for lib in ("httpx", "httpcore", "urllib3", "faiss", "sentence_transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root logging is configured first."""
    setup_logging()
    return logging.getLogger(name)


# ── Structured Request Context ─────────────────────────────────────────────────

@dataclass
class RequestContext:
    """
    Carries request-scoped metadata through the pipeline.

    Attach one to each incoming request and pass it into service functions
    so every log line can be correlated back to the same request.
    """
    request_id: str
    sender: str
    subject: str = ""
    body_length: int = 0
    thread_id: Optional[str] = None
    start_time: float = field(default_factory=time.monotonic)

    # Pipeline metrics filled in as we go
    classification_result: Optional[str] = None
    classification_confidence: Optional[float] = None
    similarity_scores: list[float] = field(default_factory=list)
    llm_latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    final_confidence: Optional[float] = None
    cache_hit: bool = False

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since this context was created."""
        return (time.monotonic() - self.start_time) * 1000

    def log_step(self, logger: logging.Logger, step: str, **extras) -> None:
        """
        Emit a structured log line for a single pipeline step.

        Args:
            logger: The calling module's logger.
            step:   Name of the pipeline step (ingestion, embedding, retrieval, …).
            extras: Key-value pairs to include in the log line.
        """
        parts = [f"request_id={self.request_id}", f"step={step}"]
        parts += [f"{k}={v}" for k, v in extras.items()]
        logger.info(" | ".join(parts))

    def log_summary(self, logger: logging.Logger) -> None:
        """
        Emit a final summary line with all collected metrics.
        Call this after the full pipeline completes.
        """
        avg_sim = (
            sum(self.similarity_scores) / len(self.similarity_scores)
            if self.similarity_scores else 0.0
        )
        logger.info(
            f"REQUEST_SUMMARY | "
            f"request_id={self.request_id} | "
            f"sender={self.sender!r} | "
            f"thread={self.thread_id!r} | "
            f"body_len={self.body_length} | "
            f"category={self.classification_result!r} | "
            f"cls_conf={self.classification_confidence:.3f} | "
            f"avg_sim={avg_sim:.3f} | "
            f"final_conf={self.final_confidence:.3f} | "
            f"llm_ms={self.llm_latency_ms:.1f} | "
            f"tokens={self.token_usage} | "
            f"cache_hit={self.cache_hit} | "
            f"total_ms={self.elapsed_ms:.1f}"
        )
