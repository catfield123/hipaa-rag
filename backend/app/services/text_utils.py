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
    return re.sub(r"\s+", " ", line).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
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
    return max(1, len(text.split()))


def marker_depth(marker: str | None) -> int:
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
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_citation_tokens(text: str) -> list[str]:
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
