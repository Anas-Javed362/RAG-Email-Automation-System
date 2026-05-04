"""
Email preprocessing and cleaning utilities.

Handles HTML stripping, signature removal, and whitespace normalization
so that downstream embedding and classification work on clean text.
"""

import re
from html.parser import HTMLParser
from typing import Optional
from app.core.logger import get_logger

logger = get_logger(__name__)

# Common email signature markers — stop parsing body when these are found
_SIGNATURE_MARKERS = [
    r"^--\s*$",
    r"^best regards[,.]?\s*$",
    r"^kind regards[,.]?\s*$",
    r"^regards[,.]?\s*$",
    r"^sincerely[,.]?\s*$",
    r"^thanks[,.]?\s*$",
    r"^thank you[,.]?\s*$",
    r"^cheers[,.]?\s*$",
    r"^warm regards[,.]?\s*$",
    r"^yours (truly|faithfully|sincerely)[,.]?\s*$",
    r"^sent from my (iphone|android|ios device|samsung|galaxy)",
    r"-{3,}original message-{3,}",
    r"on .+wrote:",
]

_SIGNATURE_PATTERN = re.compile(
    "|".join(_SIGNATURE_MARKERS),
    flags=re.IGNORECASE | re.MULTILINE,
)


class _HTMLStripper(HTMLParser):
    """Minimal HTML parser that collects plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(text: str) -> str:
    """
    Remove all HTML tags and decode entities from a string.

    Args:
        text: Raw HTML or mixed HTML/plain text.

    Returns:
        Plain text with HTML removed.
    """
    if not text or "<" not in text:
        return text

    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def remove_signature(text: str) -> str:
    """
    Truncate the email body at the first signature marker.

    Splits into lines and stops accumulating once a signature line is detected.

    Args:
        text: Plain-text email body.

    Returns:
        Body text with signature removed.
    """
    lines = text.splitlines()
    clean_lines: list[str] = []

    for line in lines:
        if _SIGNATURE_PATTERN.search(line.strip()):
            logger.debug("Signature marker detected — truncating body.")
            break
        clean_lines.append(line)

    return "\n".join(clean_lines)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/newlines and strip surrounding whitespace.

    Args:
        text: Any string.

    Returns:
        Whitespace-normalized string.
    """
    # Replace tabs and carriage returns
    text = text.replace("\t", " ").replace("\r", " ")
    # Collapse multiple blank lines into a single one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_email_body(raw_body: str) -> str:
    """
    Full preprocessing pipeline for an email body.

    Steps:
        1. Strip HTML tags
        2. Remove email signature
        3. Normalize whitespace

    Args:
        raw_body: Raw email body text (may contain HTML).

    Returns:
        Cleaned, normalized plain-text body.
    """
    text = strip_html(raw_body)
    text = remove_signature(text)
    text = normalize_whitespace(text)
    logger.debug(f"Cleaned body preview: {text[:120]!r}")
    return text


def extract_metadata(sender: str, subject: Optional[str], body: str) -> dict:
    """
    Extract structured metadata from email fields for downstream use.

    Args:
        sender:  Sender email address.
        subject: Email subject line (may be None).
        body:    Raw email body.

    Returns:
        Dict with fields: sender, subject, domain, has_question.
    """
    domain = sender.split("@")[-1] if "@" in sender else "unknown"
    combined = f"{subject or ''} {body}".lower()
    has_question = "?" in combined

    metadata = {
        "sender": sender,
        "subject": subject or "",
        "domain": domain,
        "has_question": has_question,
        "body_length": len(body),
    }
    logger.debug(f"Extracted metadata: {metadata}")
    return metadata
