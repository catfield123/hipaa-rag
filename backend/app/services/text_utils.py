"""Text normalization, tokenization, and citation-aware helpers for lexical retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable

from nltk.stem.snowball import SnowballStemmer


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "hipaa",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "under",
    "what",
    "which",
    "who",
    "there",
    "their",
    "them",
    "then",
    "than",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "also",
    "such",
    "any",
    "all",
    "each",
    "other",
}

NOISE_TOKENS = {"u", "s", "c", "et", "al"}
SECTION_CITATION_PATTERN = re.compile(r"(?:§\s*)?(\d{3}\.\d+)((?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))*)")
PART_PATTERN = re.compile(r"\bpart\s+(\d{3})\b", re.IGNORECASE)
SUBPART_PATTERN = re.compile(r"\bsubpart\s+([A-Z])\b", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-z][a-z0-9']*")
STEMMER = SnowballStemmer("english")


def normalize_line(line: str) -> str:
    """Collapse internal whitespace in a single line.

    Args:
        line (str): Raw line text.

    Returns:
        str: Single-space separated, stripped string.

    Raises:
        None
    """

    return re.sub(r"\s+", " ", line).strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace, hyphenation, and NBSP for downstream tokenization.

    Args:
        text (str): Arbitrary document or query text.

    Returns:
        str: Cleaned text safe for :func:`tokenize`.

    Raises:
        None
    """

    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Produce stemmed lexical tokens plus synthetic citation tokens for BM25-style search.

    Args:
        text (str): Query or passage text.

    Returns:
        list[str]: Stemmed word tokens and ``sec_*`` / ``cite_*`` / ``part_*`` / ``subpart_*`` features.

    Raises:
        None
    """

    normalized = normalize_text(text)
    tokens: list[str] = []
    tokens.extend(_extract_citation_tokens(normalized))

    for token in WORD_PATTERN.findall(normalized.lower()):
        token = token.strip("'")
        if not token:
            continue
        if token in STOP_WORDS or token in NOISE_TOKENS:
            continue
        if token.isdigit():
            continue
        if not re.search(r"[a-z]", token):
            continue
        if len(token) < 2:
            continue
        if token.isalnum() and not any(char.isalpha() for char in token):
            continue
        tokens.append(STEMMER.stem(token))

    return tokens


def estimate_token_count(text: str) -> int:
    """Rough token count for storage metrics (whitespace-split words, minimum 1).

    Args:
        text (str): Chunk or query text.

    Returns:
        int: At least ``1`` for non-empty inputs after split.

    Raises:
        None
    """

    return max(1, len(text.split()))


def marker_depth(marker: str | None) -> int:
    """Return a numeric nesting depth heuristic for a regulatory marker string.

    Args:
        marker (str | None): Marker text, with or without parentheses.

    Returns:
        int: Depth rank (higher means deeper); ``0`` when ``marker`` is empty.

    Raises:
        None
    """

    if not marker:
        return 0

    marker = marker.strip("()")
    if marker.isdigit():
        return 2
    if marker.isupper():
        return 4
    if re.fullmatch(r"[ivxlcdm]+", marker):
        return 3
    if marker.islower():
        return 1
    return 5


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate strings while keeping first occurrence order.

    Args:
        items (Iterable[str]): Arbitrary iterable of strings (e.g. chunk ids).

    Returns:
        list[str]: Unique items in encounter order.

    Raises:
        None
    """

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_citation_tokens(text: str) -> list[str]:
    """Build synthetic tokens from CFR section numbers and part/subpart mentions.

    Args:
        text (str): Normalized text from :func:`normalize_text`.

    Returns:
        list[str]: Feature tokens such as ``sec_164_312`` and ``part_164``.

    Raises:
        None
    """

    tokens: list[str] = []

    for match in SECTION_CITATION_PATTERN.finditer(text):
        section = match.group(1).replace(".", "_")
        suffix = match.group(2) or ""
        markers = re.findall(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)", suffix)
        tokens.append(f"sec_{section}")
        if markers:
            path = "_".join([section] + [marker.lower() for marker in markers])
            tokens.append(f"cite_{path}")

    for match in PART_PATTERN.finditer(text):
        tokens.append(f"part_{match.group(1)}")

    for match in SUBPART_PATTERN.finditer(text):
        tokens.append(f"subpart_{match.group(1).lower()}")

    return tokens
