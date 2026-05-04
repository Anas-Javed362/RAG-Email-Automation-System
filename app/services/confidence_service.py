"""
Multi-signal confidence fusion service.

Implements the weighted scoring formula:
    final_confidence = (0.4 × classification_score)
                     + (0.4 × retrieval_similarity_score)
                     + (0.2 × llm_self_score)

This approach is more reliable than using any single signal alone:
  - Classification confidence alone can be overconfident on ambiguous emails.
  - Similarity score reflects how well the RAG knowledge base covers this topic.
  - LLM self-score catches cases where the model generated a weak response.
"""

from dataclasses import dataclass
from app.core.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class FusedConfidence:
    """
    The result of multi-signal confidence fusion.

    Attributes:
        classification:  Raw classifier score (0.0–1.0).
        similarity:      Average FAISS similarity score (0.0–1.0).
        llm_self:        LLM's own rating of its response (0.0–1.0).
        final:           Fused weighted score (0.0–1.0).
        needs_review:    True when final < settings.confidence_threshold.
    """
    classification: float
    similarity: float
    llm_self: float
    final: float
    needs_review: bool


def fuse_confidence(
    classification_confidence: float,
    avg_similarity_score: float,
    llm_self_score: float,
) -> FusedConfidence:
    """
    Compute the final confidence score from three pipeline signals.

    Formula:
        final = (0.4 × classification) + (0.4 × similarity) + (0.2 × llm_self)

    All input values are clamped to [0.0, 1.0] before weighting.

    Args:
        classification_confidence:  Output from the classifier (rule + LLM combined).
        avg_similarity_score:       Mean cosine similarity from FAISS retrieval.
        llm_self_score:             LLM's self-rated response quality (0 if disabled).

    Returns:
        FusedConfidence with component scores and final weighted result.
    """
    settings = get_settings()

    # Clamp all inputs to valid range
    cls_score = max(0.0, min(1.0, classification_confidence))
    sim_score = max(0.0, min(1.0, avg_similarity_score))
    llm_score = max(0.0, min(1.0, llm_self_score))

    # Weighted fusion
    final = round(
        0.4 * cls_score + 0.4 * sim_score + 0.2 * llm_score,
        4
    )

    needs_review = final < settings.confidence_threshold

    result = FusedConfidence(
        classification=round(cls_score, 4),
        similarity=round(sim_score, 4),
        llm_self=round(llm_score, 4),
        final=final,
        needs_review=needs_review,
    )

    logger.info(
        f"Confidence fusion | "
        f"cls={cls_score:.3f} × 0.4 + sim={sim_score:.3f} × 0.4 + llm={llm_score:.3f} × 0.2 "
        f"= {final:.3f} | review={'YES' if needs_review else 'no'}"
    )

    return result
