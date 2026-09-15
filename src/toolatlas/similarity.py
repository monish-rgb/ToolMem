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

