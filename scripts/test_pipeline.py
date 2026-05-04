"""
CLI test script — run the full pipeline locally without starting the server.

Usage:
    # Test with a sample email interactively
    python scripts/test_pipeline.py

    # Test with a specific email from stdin
    python scripts/test_pipeline.py --sender "user@example.com" --subject "Help!" --body "I can't login"

    # Test with thread context
    python scripts/test_pipeline.py --thread-id "thread-001" --body "Still can't login, tried reset"

    # Skip LLM classification (faster for offline testing)
    python scripts/test_pipeline.py --no-llm
"""

import sys
import json
import time
import asyncio
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logger import setup_logging
setup_logging()

from config.settings import get_settings
from ingestion.email_cleaner import clean_email_body
from rag.embedder import embed_text, get_embedding_dimension
from rag.retriever import retrieve_similar_emails, build_context_string, get_average_retrieval_score
from classifiers.classifier import classify_email
from rag.vector_store import get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the RAG Email pipeline locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sender", default="test@example.com", help="Sender email address")
    parser.add_argument("--subject", default="Test email", help="Email subject")
    parser.add_argument("--body", default=None, help="Email body text (prompts interactively if omitted)")
    parser.add_argument("--thread-id", default=None, help="Thread ID for conversation context")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (rule-based classifier only)")
    parser.add_argument("--prompt-version", default=None, help="Override prompt version (v1 or v2)")
    return parser.parse_args()


def print_section(title: str, content: str = "") -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    if content:
        print(content)


def check_index_health() -> bool:
    """Verify that the FAISS index has been seeded."""
    store = get_vector_store()
    if store.total_vectors == 0:
        print("\n⚠️  WARNING: FAISS index is empty.")
        print("   Run: python scripts/seed_data.py")
        print("   Retrieval will return no results.\n")
        return False
    return True


async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline and display results."""
    settings = get_settings()

    # Get email body
    if args.body:
        body = args.body
    else:
        print("\n📧 Enter email body (press Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        body = "\n".join(lines)

    if not body.strip():
        print("❌ No email body provided. Exiting.")
        sys.exit(1)

    print_section("INPUT EMAIL")
    print(f"  Sender  : {args.sender}")
    print(f"  Subject : {args.subject}")
    print(f"  Thread  : {args.thread_id or '(standalone)'}")
    print(f"  Body    : {body[:200]}{'...' if len(body) > 200 else ''}")

    t_start = time.monotonic()

    # Step 1: Clean
    print_section("STEP 1: PREPROCESSING")
    cleaned = clean_email_body(body)
    print(f"  Cleaned body ({len(cleaned)} chars):")
    print(f"  {cleaned[:300]}{'...' if len(cleaned) > 300 else ''}")

    # Step 2: Classify
    print_section("STEP 2: CLASSIFICATION")
    use_llm = not args.no_llm
    if not use_llm:
        print("  [LLM disabled — using rule-based only]")
    classification = classify_email(cleaned, use_llm=use_llm)
    print(f"  Category   : {classification.category}")
    print(f"  Confidence : {classification.confidence:.3f}")
    print(f"  Method     : {classification.method}")

    # Step 3: Retrieve
    print_section("STEP 3: RETRIEVAL (FAISS)")
    check_index_health()
    similar = retrieve_similar_emails(cleaned)
    avg_sim = get_average_retrieval_score(similar)
    print(f"  Retrieved  : {len(similar)} similar emails")
    print(f"  Avg score  : {avg_sim:.3f}")
    for i, r in enumerate(similar, 1):
        print(f"  [{i}] [{r.category or '?':10s}] score={r.score:.3f} | {r.body_cleaned[:80]}...")

    context_str = build_context_string(similar)

    # Step 4: Generate (async)
    print_section("STEP 4: GENERATION (LLM)")
    if args.no_llm:
        print("  [LLM disabled — skipping generation]")
        response_text = "[LLM generation skipped in --no-llm mode]"
        llm_self_score = 0.5
        latency_ms = 0.0
    else:
        from rag.generator import generate_response
        gen = await generate_response(
            email_body=cleaned,
            sender=args.sender,
            subject=args.subject,
            category=classification.category,
            context=context_str,
            thread_history="",
            prompt_version=args.prompt_version or settings.prompt_version,
        )
        response_text = gen.response_text
        llm_self_score = gen.llm_self_score
        latency_ms = gen.latency_ms
        print(f"  LLM latency    : {latency_ms:.0f}ms")
        print(f"  Tokens used    : {gen.token_usage}")
        print(f"  LLM self-score : {llm_self_score:.3f}")
        print(f"  Prompt version : {gen.prompt_version}")

    # Step 5: Confidence fusion
    print_section("STEP 5: CONFIDENCE FUSION")
    from app.services.confidence_service import fuse_confidence
    fused = fuse_confidence(classification.confidence, avg_sim, llm_self_score)
    print(f"  Classification : {fused.classification:.3f} × 0.4")
    print(f"  Similarity     : {fused.similarity:.3f} × 0.4")
    print(f"  LLM self-score : {fused.llm_self:.3f} × 0.2")
    print(f"  ─────────────────────────────")
    print(f"  Final score    : {fused.final:.3f}")
    print(f"  Needs review   : {'⚠️  YES' if fused.needs_review else '✅ No'}")

    # Final output
    total_ms = (time.monotonic() - t_start) * 1000
    print_section("📬 GENERATED RESPONSE")
    print(f"\n{response_text}\n")
    print_section("📊 SUMMARY")
    print(f"  Category      : {classification.category}")
    print(f"  Confidence    : {fused.final:.3f}")
    print(f"  Needs review  : {'YES ⚠️' if fused.needs_review else 'No ✅'}")
    print(f"  Total latency : {total_ms:.0f}ms")
    print()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(args))
