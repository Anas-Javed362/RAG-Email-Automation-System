"""
Batch evaluation script — tests the pipeline against the seed dataset.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output reports/eval_result.json
    python scripts/evaluate.py --no-llm

Computes:
    - Per-category classification accuracy
    - Average response time
    - % auto-resolved vs needs human review
    - Average similarity score
    - Confidence distribution
"""

import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logger import setup_logging
setup_logging()

from config.settings import get_settings
from ingestion.email_cleaner import clean_email_body
from classifiers.classifier import classify_email
from rag.retriever import retrieve_similar_emails, get_average_retrieval_score
from app.services.confidence_service import fuse_confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation of the RAG email pipeline.")
    parser.add_argument("--dataset", default="data/seed_emails.json", help="Path to evaluation dataset")
    parser.add_argument("--output", default=None, help="Optional path to save JSON report")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM classification")
    return parser.parse_args()


def compute_metrics(results: list[dict]) -> dict:
    """
    Aggregate evaluation results into a structured metrics report.

    Args:
        results: List of per-email result dicts from the evaluation run.

    Returns:
        Dict with accuracy, latency, review rate, and confidence stats.
    """
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total

    latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(latencies) / total
    max_latency = max(latencies)
    min_latency = min(latencies)

    review_needed = sum(1 for r in results if r["needs_review"])
    review_rate = review_needed / total

    confidences = [r["confidence"] for r in results]
    avg_confidence = sum(confidences) / total

    sim_scores = [r["avg_sim"] for r in results]
    avg_sim = sum(sim_scores) / total

    # Per-category breakdown
    per_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        true_cat = r["true_category"]
        per_cat[true_cat]["total"] += 1
        if r["correct"]:
            per_cat[true_cat]["correct"] += 1

    cat_accuracy = {
        cat: round(v["correct"] / v["total"], 3)
        for cat, v in per_cat.items()
    }

    return {
        "total_emails": total,
        "overall_accuracy": round(accuracy, 3),
        "per_category_accuracy": cat_accuracy,
        "avg_latency_ms": round(avg_latency, 1),
        "max_latency_ms": round(max_latency, 1),
        "min_latency_ms": round(min_latency, 1),
        "avg_confidence": round(avg_confidence, 3),
        "avg_similarity": round(avg_sim, 3),
        "auto_resolved_count": total - review_needed,
        "human_review_count": review_needed,
        "auto_resolved_pct": round((total - review_needed) / total * 100, 1),
        "human_review_pct": round(review_rate * 100, 1),
    }


async def evaluate_single(email: dict, use_llm: bool) -> dict:
    """
    Run the pipeline on a single email and collect metrics.

    Args:
        email:   Dict with sender, subject, body, category fields.
        use_llm: Whether to use LLM classification.

    Returns:
        Dict with metrics for this email.
    """
    true_category = email.get("category", "Unknown")
    body = email.get("body", "")

    t0 = time.monotonic()

    # Clean
    cleaned = clean_email_body(body)

    # Classify
    classification = await asyncio.get_event_loop().run_in_executor(
        None, lambda: classify_email(cleaned, use_llm=use_llm)
    )

    # Retrieve
    similar = await asyncio.get_event_loop().run_in_executor(
        None, lambda: retrieve_similar_emails(cleaned)
    )
    avg_sim = get_average_retrieval_score(similar)

    # Confidence fusion (no LLM self-eval during batch eval — too slow)
    fused = fuse_confidence(
        classification_confidence=classification.confidence,
        avg_similarity_score=avg_sim,
        llm_self_score=0.75,  # Neutral for batch evaluation
    )

    latency_ms = (time.monotonic() - t0) * 1000

    return {
        "sender": email.get("sender", ""),
        "subject": email.get("subject", ""),
        "true_category": true_category,
        "predicted_category": classification.category,
        "correct": classification.category == true_category,
        "classification_confidence": round(classification.confidence, 3),
        "avg_sim": round(avg_sim, 3),
        "confidence": round(fused.final, 3),
        "needs_review": fused.needs_review,
        "latency_ms": round(latency_ms, 1),
        "method": classification.method,
    }


async def run_evaluation(args: argparse.Namespace) -> None:
    """Main evaluation runner."""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        emails = json.load(f)

    settings = get_settings()
    use_llm = not args.no_llm

    print(f"\n{'═' * 60}")
    print(f"  RAG Email System — Batch Evaluation")
    print(f"{'═' * 60}")
    print(f"  Dataset       : {dataset_path} ({len(emails)} emails)")
    print(f"  LLM enabled   : {use_llm}")
    print(f"  Prompt version: {settings.prompt_version}")
    print(f"  Threshold     : {settings.confidence_threshold}")
    print(f"{'─' * 60}\n")

    results = []
    for i, email in enumerate(emails, 1):
        subject = email.get("subject", "")[:50]
        print(f"  [{i:02d}/{len(emails)}] {subject:50s}", end=" ", flush=True)

        result = await evaluate_single(email, use_llm=use_llm)
        results.append(result)

        status = "✅" if result["correct"] else "❌"
        review = "⚠️" if result["needs_review"] else "  "
        print(
            f"{status} {review} "
            f"pred={result['predicted_category']:10s} "
            f"conf={result['confidence']:.2f} "
            f"{result['latency_ms']:.0f}ms"
        )

    metrics = compute_metrics(results)

    print(f"\n{'═' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'═' * 60}")
    print(f"  Overall Accuracy    : {metrics['overall_accuracy']:.1%}")
    print(f"  Avg Confidence      : {metrics['avg_confidence']:.3f}")
    print(f"  Avg Similarity      : {metrics['avg_similarity']:.3f}")
    print(f"  Avg Latency         : {metrics['avg_latency_ms']:.0f}ms")
    print(f"  Auto Resolved       : {metrics['auto_resolved_count']}/{metrics['total_emails']} ({metrics['auto_resolved_pct']}%)")
    print(f"  Needs Human Review  : {metrics['human_review_count']}/{metrics['total_emails']} ({metrics['human_review_pct']}%)")
    print(f"\n  Per-Category Accuracy:")
    for cat, acc in sorted(metrics["per_category_accuracy"].items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {cat:12s} {bar} {acc:.1%}")
    print(f"{'═' * 60}\n")

    # Save JSON report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "llm_enabled": use_llm,
            "prompt_version": settings.prompt_version,
            "confidence_threshold": settings.confidence_threshold,
        },
        "metrics": metrics,
        "per_email_results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"📊 Report saved to: {output_path}")
    else:
        print(json.dumps({"metrics": metrics}, indent=2))


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_evaluation(args))
