from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> list[str]:
    """Cheap, local embedding substitute suitable for a minimal prototype."""
    return TOKEN_RE.findall(text.lower().replace("-", "_"))


def cosine_text(left: str, right: str) -> float:
    a, b = Counter(tokens(left)), Counter(tokens(right))
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


SYNONYMS = {
    "tally": "count",
    "find": "search",
    "locate": "search",
    "lookup": "search",
    "clean": "normalize",
    "tidy": "normalize",
    "divide": "split",
    "combine": "merge",
    "join": "merge",
    "organize": "move",
    "inspect": "read",
    "examine": "read",
    "make": "create",
    "build": "create",
    "show": "list",
    "display": "list",
    "big": "large",
    "tiny": "small",
    "little": "small",
    "remove": "delete",
    "erase": "delete",
    "compute": "calculate",
}


def normalized_tokens(text: str) -> list[str]:
    """Tokens with a small curated synonym map for paraphrase recall."""
    return [SYNONYMS.get(token, token) for token in tokens(text)]


def trigrams(text: str) -> set[str]:
    """Character-trigram set over whitespace-collapsed lowercase text."""
    collapsed = " ".join(text.lower().split())
    if len(collapsed) < 3:
        return set()
    return {collapsed[index:index + 3] for index in range(len(collapsed) - 2)}


def trigram_similarity(left: str, right: str) -> float:
    """Jaccard similarity over character trigrams; 0.0 when incomparable."""
    left_set, right_set = trigrams(left), trigrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def hybrid_similarity(left: str, right: str) -> float:
    """Lexical cosine with a trigram backoff for paraphrased queries."""
    cosine = cosine_text(left, right)
    if cosine > 0:
        return cosine
    return trigram_similarity(left, right)

