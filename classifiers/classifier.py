"""
Email classification module.

Implements two complementary strategies:
  1. Rule-based: Keyword matching with weighted scoring — fast, explainable, no API cost.
  2. LLM-based:  Zero-shot classification via LangChain — smarter for ambiguous emails.

Results are merged using a weighted average to produce a final confidence score.
"""

import re
from dataclasses import dataclass
from typing import Optional

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Output from the classifier."""
    category: str
    confidence: float
    method: str  # 'rule_based', 'llm', or 'combined'


# ── Keyword taxonomy for rule-based classification ─────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Complaint": [
        "broken", "damaged", "defective", "not working", "disappointed",
        "terrible", "awful", "horrible", "unacceptable", "refund", "return",
        "complaint", "issue", "problem", "fault", "error", "frustrated",
        "angry", "worst", "never again", "scam", "fraud", "poor quality",
    ],
    "Inquiry": [
        "how do i", "can you tell me", "what is", "when will", "where is",
        "i would like to know", "could you please", "is it possible",
        "do you offer", "what are the", "information about", "curious",
        "question", "wondering", "inquiry", "enquiry", "price", "availability",
        "hours", "location", "details",
    ],
    "Support": [
        "help", "assist", "support", "guide", "stuck", "cannot", "can't",
        "unable", "not able", "doesn't work", "how to", "steps",
        "configure", "setup", "install", "reset", "password", "login",
        "technical", "access", "trouble", "fix", "resolve",
    ],
    "Spam": [
        "congratulations you've won", "click here", "limited time offer",
        "free gift", "act now", "exclusive deal", "unsubscribe",
        "earn money", "make money online", "work from home", "100% free",
        "no obligation", "winner", "lottery", "prize", "guaranteed",
        "lose weight", "special promotion", "buy now", "order now",
    ],
}


def _rule_based_classify(text: str) -> ClassificationResult:
    """
    Classify email using keyword frequency scoring.

    Each keyword match increases the score for its category.
    The category with the most matches wins. Ties break alphabetically.

    Args:
        text: Cleaned email body (lowercased internally).

    Returns:
        ClassificationResult with category and a confidence between 0.0 and 1.0.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {category: 0 for category in _CATEGORY_KEYWORDS}

    for category, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[category] += 1

    total_matches = sum(scores.values())

    if total_matches == 0:
        logger.debug("Rule-based: no keyword matches found — defaulting to Inquiry.")
        return ClassificationResult(category="Inquiry", confidence=0.3, method="rule_based")

    best_category = max(scores, key=lambda c: (scores[c], c))
    confidence = min(scores[best_category] / max(total_matches, 1), 1.0)
    # Scale confidence: 100% of matches → 0.95, partial → proportionally less
    confidence = 0.4 + 0.55 * confidence

    logger.debug(f"Rule-based scores: {scores} → {best_category} ({confidence:.2f})")
    return ClassificationResult(
        category=best_category,
        confidence=round(confidence, 3),
        method="rule_based",
    )


def _llm_based_classify(text: str) -> Optional[ClassificationResult]:
    """
    Classify email using zero-shot LLM prompting via LangChain.

    Falls back to None if LLM is unavailable or call fails.

    Args:
        text: Cleaned email body.

    Returns:
        ClassificationResult or None on failure.
    """
    settings = get_settings()

    try:
        from langchain.schema import HumanMessage, SystemMessage

        if settings.is_openai:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
        elif settings.is_huggingface:
            from langchain_community.llms import HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id=settings.huggingface_model,
                huggingfacehub_api_token=settings.huggingface_api_token,
                model_kwargs={"temperature": 0.1, "max_new_tokens": 50},
            )
        else:
            logger.warning(f"Unknown LLM provider: {settings.llm_provider}")
            return None

        system_prompt = (
            "You are an expert email classifier. Classify the following email into exactly "
            "one of these categories: Complaint, Inquiry, Support, Spam.\n"
            "Respond with a JSON object in this exact format (no extra text):\n"
            '{"category": "<Category>", "confidence": <0.0-1.0>}'
        )

        human_prompt = f"Email body:\n\n{text[:1500]}"  # Truncate to avoid token limits

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse the JSON response
        import json
        match = re.search(r'\{.*?"category".*?\}', response_text, re.DOTALL)
        if not match:
            logger.warning(f"LLM classifier: could not parse response: {response_text[:200]}")
            return None

        parsed = json.loads(match.group())
        category = parsed.get("category", "Inquiry")
        confidence = float(parsed.get("confidence", 0.75))

        # Validate category
        valid_categories = {"Complaint", "Inquiry", "Support", "Spam"}
        if category not in valid_categories:
            logger.warning(f"LLM returned unknown category '{category}' — defaulting to Inquiry")
            category = "Inquiry"

        logger.info(f"LLM classifier: {category} ({confidence:.2f})")
        return ClassificationResult(category=category, confidence=confidence, method="llm")

    except Exception as exc:
        logger.warning(f"LLM classification failed ({type(exc).__name__}: {exc}). Falling back to rule-based.")
        return None


def classify_email(text: str, use_llm: bool = True) -> ClassificationResult:
    """
    Classify an email using the combined rule-based + LLM approach.

    Strategy:
        - Always run rule-based (fast, fallback)
        - Attempt LLM classification if `use_llm=True` and configuration is available
        - If both succeed and agree → high confidence
        - If both succeed but disagree → trust LLM, moderate confidence
        - If only rule-based → use its score

    Args:
        text:     Cleaned email body text.
        use_llm:  Whether to attempt LLM classification.

    Returns:
        ClassificationResult with final category, confidence, and method.
    """
    rule_result = _rule_based_classify(text)

    if not use_llm:
        return rule_result

    llm_result = _llm_based_classify(text)

    if llm_result is None:
        # LLM unavailable — use rule-based only
        logger.info(f"Classification (rule-based only): {rule_result.category} ({rule_result.confidence:.2f})")
        return rule_result

    # Merge results
    if rule_result.category == llm_result.category:
        # Both agree → boost confidence
        combined_confidence = min(0.9 * llm_result.confidence + 0.1 * rule_result.confidence + 0.05, 1.0)
        method = "combined_agree"
    else:
        # Disagree → trust LLM more (70/30 weighting)
        combined_confidence = 0.7 * llm_result.confidence + 0.3 * rule_result.confidence
        method = "combined_disagree"
        logger.info(
            f"Classifier disagreement: rule={rule_result.category}, llm={llm_result.category} "
            f"→ using LLM: {llm_result.category}"
        )

    final = ClassificationResult(
        category=llm_result.category,
        confidence=round(combined_confidence, 3),
        method=method,
    )
    logger.info(f"Final classification: {final.category} ({final.confidence:.2f}) via {final.method}")
    return final
