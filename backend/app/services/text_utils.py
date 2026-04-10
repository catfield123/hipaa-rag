from __future__ import annotations

import re
from collections.abc import Iterable


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
}


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


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
