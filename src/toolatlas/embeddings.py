"""Versioned embedding similarity for retrieval (Phase 4).

The lexical/trigram retriever stays the deterministic fallback. This module
adds the paper-aligned path: task and trace summaries compared by embedding
cosine, with an explicit ``(model, version)`` recorded on every comparison.
Only same-version vectors are ever compared — a model/version change
invalidates stored vectors and retrieval falls back to lexical, never mixing
incomparable spaces.

- ``HashEmbedder``: deterministic, credential-free, offline. Used in tests,
  the offline demo, and anywhere frozen memory is served without network.
- ``ProviderEmbedder``: live Gemini / OpenAI-compatible embeddings behind an
  explicit constructor. Never built implicitly; tests never touch it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .similarity import normalized_tokens

HASH_MODEL = "hash"
HASH_VERSION = "hash-v1"
HASH_DIM = 256


def cosine_vec(left: list[float], right: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for empty or mismatched vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class Embedder(Protocol):
    """Text embedder with explicit model/version identity."""

    model: str
    version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _signed_hashes(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode()).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


@dataclass
class HashEmbedder:
    """Deterministic hashing-trick embedder over synonym-normalized tokens.

    Credential-free and stable across processes: identical text always yields
    an identical unit vector, so frozen evaluation memory can serve it with
    no network and no stored vectors.
    """

    model: str = HASH_MODEL
    version: str = HASH_VERSION
    dim: int = HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in normalized_tokens(text or ""):
                index, sign = _signed_hashes(f"{self.version}:{token}", self.dim)
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


@dataclass
class ProviderEmbedder:
    """Live provider embeddings. Explicit opt-in only (paid, networked)."""

    api_key: str
    model: str
    provider: str = "gemini"  # or "openai_compatible"
    base_url: str = ""
    version: str = ""

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("ProviderEmbedder requires an explicit api_key")
        if not self.model:
            raise RuntimeError("ProviderEmbedder requires an explicit model")
        if not self.version:
            self.version = f"{self.provider}:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "gemini":
            return [self._gemini_embed(text) for text in texts]
        return self._openai_embed(texts)

    def _gemini_embed(self, text: str) -> list[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        body = json.dumps({"content": {"parts": [{"text": text or " "}]}}).encode()
        request = urllib.request.Request(
            url, data=body,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode())
        values = (data.get("embedding") or {}).get("values") or []
        if not values:
            raise RuntimeError(f"Gemini embedding returned no values for model {self.model}")
        return [float(value) for value in values]

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        base = (self.base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        if not base:
            raise RuntimeError("openai_compatible embeddings require a base_url")
        body = json.dumps({"input": texts, "model": self.model}).encode()
        request = urllib.request.Request(
            f"{base}/embeddings", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode())
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        if len(items) != len(texts):
            raise RuntimeError("embedding count mismatch from provider")
        return [[float(value) for value in item["embedding"]] for item in items]


def provider_embedder_from_env(model: str = "", provider: str = "") -> ProviderEmbedder:
    """Explicit live-embedder factory. Raises when no credential is configured."""
    from .gemini_rest import provider_name
    resolved_provider = (provider or provider_name())
    if resolved_provider == "gemini":
        key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY", "")).strip().strip('"').strip("'")
        resolved_model = (model or os.environ.get("LLM_EMBED_MODEL", "") or "text-embedding-004").strip()
        if not key:
            raise RuntimeError("set $env:GEMINI_API_KEY (or $env:LLM_API_KEY) for provider embeddings")
        return ProviderEmbedder(api_key=key, model=resolved_model, provider="gemini")
    key = (os.environ.get("LLM_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("set $env:LLM_API_KEY for provider embeddings")
    resolved_model = (model or os.environ.get("LLM_EMBED_MODEL", "")).strip()
    if not resolved_model:
        raise RuntimeError("set $env:LLM_EMBED_MODEL for provider embeddings")
    return ProviderEmbedder(api_key=key, model=resolved_model,
                            provider="openai_compatible",
                            base_url=os.environ.get("LLM_BASE_URL", ""))


def embedding_similarity(query_vec: list[float],
                         candidate_vecs: list[list[float]]) -> list[float]:
    """Cosine of the query against each candidate (same-version only)."""
    return [cosine_vec(query_vec, candidate) for candidate in candidate_vecs]


@dataclass(slots=True)
class StoredEmbedding:
    """One cached provider vector with its comparability identity."""

    qid: str
    model: str
    version: str
    vector: list[float]


def vectors_compatible(stored: StoredEmbedding | None, embedder: Embedder) -> bool:
    """Same model and version, non-empty vector — else lexical fallback."""
    return bool(stored and stored.vector
                and stored.model == embedder.model
                and stored.version == embedder.version)


def _demo() -> None:  # pragma: no cover - manual inspection helper
    embedder = HashEmbedder()
    [query], [near, far] = embedder.embed(["Classify files by size"]), embedder.embed(
        ["Classify files by size and move them", "Bake sourdough bread slowly"])
    print({"near": cosine_vec(query, near), "far": cosine_vec(query, far)})


if __name__ == "__main__":  # pragma: no cover
    _demo()
