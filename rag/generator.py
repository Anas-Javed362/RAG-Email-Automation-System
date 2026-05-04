"""
LLM response generator with:
  - Prompt versioning (v1 / v2 selectable via settings)
  - Exponential backoff retry on API failures
  - LLM self-evaluation (rates its own response confidence)
  - Support for OpenAI and HuggingFace backends

This is the most I/O-bound step in the pipeline, so it includes
careful error handling and retry logic to survive transient API issues.
"""

import asyncio
import time
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Output from the LLM generator."""
    response_text: str
    llm_self_score: float    # LLM's own confidence rating (0.0–1.0)
    latency_ms: float
    token_usage: dict        # {"prompt_tokens": n, "completion_tokens": n, "total": n}
    prompt_version: str


# ── Prompt Loading ─────────────────────────────────────────────────────────────

def _load_prompt_template(version: str) -> str:
    """
    Load a prompt template from the rag/prompts/ directory.

    Args:
        version: Template version string, e.g. 'v1' or 'v2'.

    Returns:
        Raw template string with {placeholder} variables.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    settings = get_settings()
    template_path = Path(settings.prompts_dir) / f"{version}.txt"

    if not template_path.exists():
        logger.warning(f"Prompt template {template_path} not found — falling back to v1.")
        template_path = Path(settings.prompts_dir) / "v1.txt"

    text = template_path.read_text(encoding="utf-8")
    logger.debug(f"Loaded prompt template: {template_path} ({len(text)} chars)")
    return text


def _build_prompt(
    email_body: str,
    sender: str,
    subject: str,
    category: str,
    context: str,
    thread_history: str = "",
    version: Optional[str] = None,
) -> str:
    """
    Fill in the prompt template with actual values.

    Args:
        email_body:     Cleaned email body text.
        sender:         Email sender address.
        subject:        Email subject (may be empty).
        category:       Predicted category (Complaint, Inquiry, etc.).
        context:        Formatted similar emails from FAISS retrieval.
        thread_history: Formatted conversation history (for v2 template).
        version:        Prompt version to use. Defaults to settings.prompt_version.

    Returns:
        Fully filled prompt string ready to send to the LLM.
    """
    settings = get_settings()
    ver = version or settings.prompt_version
    template = _load_prompt_template(ver)

    filled = template.format(
        email_body=email_body,
        sender=sender,
        subject=subject or "(no subject)",
        category=category,
        context=context,
        thread_history=thread_history or "No prior conversation history.",
    )
    return filled


# ── LLM Client Factory ─────────────────────────────────────────────────────────

def _get_llm_client():
    """
    Build and return the appropriate LLM client based on settings.

    Returns:
        A LangChain chat model or LLM instance.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    settings = get_settings()

    if settings.is_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0.4,
            openai_api_key=settings.openai_api_key,
            request_timeout=settings.llm_timeout_seconds,
        )

    if settings.is_huggingface:
        from langchain_community.llms import HuggingFaceHub
        return HuggingFaceHub(
            repo_id=settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_api_token,
            model_kwargs={"temperature": 0.4, "max_new_tokens": 300},
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")


# ── Retry Decorator ────────────────────────────────────────────────────────────

async def _call_llm_with_retry(prompt: str) -> tuple[str, dict]:
    """
    Call the LLM with exponential backoff retry logic.

    Handles:
      - openai.RateLimitError → waits and retries
      - openai.APITimeoutError → retries with increasing delay
      - Generic exceptions → retries up to max_retries times

    Args:
        prompt: The fully built prompt string.

    Returns:
        Tuple of (response_text, token_usage_dict).

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    settings = get_settings()
    max_retries = settings.llm_max_retries
    base_delay = settings.llm_retry_base_delay

    llm = _get_llm_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"LLM call attempt {attempt}/{max_retries}")
            t0 = time.monotonic()

            # Run the blocking LLM call in a thread so we don't block the event loop
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: llm.invoke(prompt)
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            # Extract text content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # Extract token usage if available (OpenAI only)
            token_usage = {}
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata
                usage = meta.get("token_usage", {})
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                }

            logger.info(f"LLM responded in {elapsed_ms:.0f}ms | tokens={token_usage}")
            return response_text.strip(), token_usage

        except Exception as exc:
            last_error = exc
            error_type = type(exc).__name__

            # Check if this is a rate limit error
            is_rate_limit = "rate" in str(exc).lower() or "429" in str(exc)
            is_timeout = "timeout" in str(exc).lower()

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                if is_rate_limit:
                    delay = max(delay, 5.0)  # At least 5s for rate limits
                logger.warning(
                    f"LLM call failed ({error_type}: {exc}) — "
                    f"retry {attempt}/{max_retries} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts: {exc}")

    raise RuntimeError(f"LLM generation failed after {max_retries} retries: {last_error}")


# ── LLM Self-Evaluation ────────────────────────────────────────────────────────

async def _evaluate_response_confidence(
    original_email: str,
    generated_response: str,
) -> float:
    """
    Ask the LLM to rate the quality/confidence of its own response.

    This self-reflective score is used as the third signal in the confidence fusion:
        final = 0.4*cls + 0.4*similarity + 0.2*llm_self_score

    Args:
        original_email:     The cleaned email body that was processed.
        generated_response: The response the LLM generated.

    Returns:
        A float in [0.0, 1.0]. Returns 0.7 on failure (neutral default).
    """
    eval_prompt = (
        f"Original email:\n{original_email[:800]}\n\n"
        f"Generated response:\n{generated_response}\n\n"
        "On a scale from 0.0 to 1.0, rate how confident you are that the above response "
        "is accurate, helpful, and appropriate for the original email.\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."
    )

    try:
        raw_text, _ = await _call_llm_with_retry(eval_prompt)
        # Extract the first float-like token from the response
        import re
        match = re.search(r"0?\.\d+|1\.0|1", raw_text.strip())
        if match:
            score = float(match.group())
            score = max(0.0, min(1.0, score))
            logger.info(f"LLM self-eval score: {score:.2f}")
            return score
        logger.warning(f"Could not parse LLM self-eval response: {raw_text!r}")
        return 0.7
    except Exception as exc:
        logger.warning(f"LLM self-evaluation failed ({exc}) — using neutral score 0.7")
        return 0.7


# ── Main Generator ─────────────────────────────────────────────────────────────

async def generate_response(
    email_body: str,
    sender: str,
    subject: str,
    category: str,
    context: str,
    thread_history: str = "",
    prompt_version: Optional[str] = None,
) -> GenerationResult:
    """
    Generate a professional email response using the configured LLM.

    Pipeline:
        1. Load versioned prompt template
        2. Fill in all variables
        3. Call LLM with retry + backoff
        4. Optionally request LLM self-evaluation score
        5. Return GenerationResult with text, score, and metadata

    Args:
        email_body:     Cleaned email body for context.
        sender:         Sender email address (personalization).
        subject:        Email subject line.
        category:       Predicted category (sets tone in v2 prompt).
        context:        FAISS-retrieved similar emails (formatted string).
        thread_history: Prior conversation messages (formatted string, v2 only).
        prompt_version: Override the settings.prompt_version if needed.

    Returns:
        GenerationResult with response text, self-eval score, and token usage.
    """
    settings = get_settings()
    ver = prompt_version or settings.prompt_version

    logger.info(f"Generating response | category={category} | prompt={ver} | thread_history={'yes' if thread_history else 'no'}")

    prompt = _build_prompt(
        email_body=email_body,
        sender=sender,
        subject=subject,
        category=category,
        context=context,
        thread_history=thread_history,
        version=ver,
    )

    t0 = time.monotonic()
    response_text, token_usage = await _call_llm_with_retry(prompt)
    latency_ms = (time.monotonic() - t0) * 1000

    # LLM self-evaluation (optional — controlled by settings)
    if settings.enable_llm_self_eval:
        llm_self_score = await _evaluate_response_confidence(email_body, response_text)
    else:
        llm_self_score = 0.75  # Skip self-eval — use a reasonable neutral default

    return GenerationResult(
        response_text=response_text,
        llm_self_score=llm_self_score,
        latency_ms=latency_ms,
        token_usage=token_usage,
        prompt_version=ver,
    )
