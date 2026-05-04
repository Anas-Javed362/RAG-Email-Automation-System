"""
Seed the FAISS vector store with sample emails from data/seed_emails.json.

Run once before starting the server:
    python scripts/seed_data.py

Options:
    --reset    Rebuild the FAISS index from scratch (deletes existing data)
    --verbose  Show embedding progress
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from app.core.logger import setup_logging
from ingestion.email_cleaner import clean_email_body
from rag.embedder import embed_batch, get_embedding_dimension
from rag.vector_store import FaissVectorStore

setup_logging()
logger = logging.getLogger("seed_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed FAISS index with sample emails.")
    parser.add_argument("--reset", action="store_true", help="Delete existing index before seeding.")
    parser.add_argument("--verbose", action="store_true", help="Print each email as it is seeded.")
    return parser.parse_args()


def load_seed_emails(json_path: Path) -> list[dict]:
    """Load and validate the seed email JSON file."""
    if not json_path.exists():
        logger.error(f"Seed file not found: {json_path}")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        emails = json.load(f)
    logger.info(f"Loaded {len(emails)} seed emails from {json_path}")
    return emails


def seed(reset: bool = False, verbose: bool = False) -> None:
    """
    Main seeding routine.

    Steps:
        1. Load seed emails from JSON
        2. Clean each email body
        3. Batch-embed all bodies in one shot
        4. Add all vectors to FAISS with metadata
        5. Save index to disk

    Args:
        reset:   If True, delete the existing index before seeding.
        verbose: If True, print each email's details.
    """
    settings = get_settings()
    seed_path = Path("data/seed_emails.json")
    index_path = settings.faiss_index_path

    # Reset existing index if requested
    if reset:
        for ext in [".idx", ".meta.pkl"]:
            p = Path(index_path).with_suffix(ext)
            if p.exists():
                p.unlink()
                logger.info(f"Deleted existing index file: {p}")

    emails = load_seed_emails(seed_path)

    # Clean all email bodies
    print("\n📧 Cleaning email bodies...")
    cleaned_bodies: list[str] = []
    metadatas: list[dict] = []

    for i, email in enumerate(emails, start=1):
        raw_body = email.get("body", "")
        cleaned = clean_email_body(raw_body)
        cleaned_bodies.append(cleaned)
        metadatas.append({
            "body_cleaned": cleaned,
            "category": email.get("category", "Unknown"),
            "sender": email.get("sender", ""),
            "subject": email.get("subject", ""),
        })
        if verbose:
            print(f"  [{i:02d}] [{email.get('category', '?'):10s}] {email.get('subject', '')[:60]}")

    # Batch embed
    print(f"\n🔢 Generating embeddings for {len(cleaned_bodies)} emails...")
    vectors = embed_batch(cleaned_bodies)
    print(f"   Embedding shape: {vectors.shape}")

    # Build or load FAISS store
    dim = get_embedding_dimension()
    store = FaissVectorStore(dimension=dim, index_path=index_path)

    # Add all at once
    print("\n💾 Storing vectors in FAISS...")
    ids = store.add_batch(vectors, metadatas)
    print(f"   Assigned vector IDs: {ids[0]}–{ids[-1]}")

    # Category distribution summary
    categories = [m["category"] for m in metadatas]
    from collections import Counter
    dist = Counter(categories)

    print("\n✅ Seeding complete!")
    print(f"   Total vectors in index: {store.total_vectors}")
    print(f"   Category distribution:")
    for cat, count in sorted(dist.items()):
        print(f"     {cat:12s}: {count}")
    print(f"\n   Index saved to: {index_path}\n")


if __name__ == "__main__":
    args = parse_args()
    seed(reset=args.reset, verbose=args.verbose)
