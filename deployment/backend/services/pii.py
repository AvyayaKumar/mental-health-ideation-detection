import re
from typing import Tuple, Dict

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")

REPLACEMENTS = {
    EMAIL_RE: "[EMAIL]",
    PHONE_RE: "[PHONE]",
}


def anonymize(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Basic PII anonymization: replace emails and phone numbers.
    Returns redacted text and counts by placeholder type.
    """
    counts: Dict[str, int] = {"[EMAIL]": 0, "[PHONE]": 0}
    redacted = text
    for pattern, placeholder in REPLACEMENTS.items():
        redacted, n = pattern.subn(placeholder, redacted)
        counts[placeholder] += n
    return redacted, counts
