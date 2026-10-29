"""Embedding-model registry and a thin encoder wrapper.

All real models are loaded through sentence-transformers. E5 models need the
"query: " / "passage: " prefixes; forgetting them silently costs several MRR points,
so prefixes live in the registry next to the model id.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int | None = 512  # None -> keep the model's own default


_SPECS = [
    ModelSpec("e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: "),
    ModelSpec("e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    ModelSpec("minilm-l12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_seq_length=None),
    ModelSpec("bge-m3", "BAAI/bge-m3"),
    # optional extras (not in the default comparison)
    ModelSpec("e5-large", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    ModelSpec("labse", "sentence-transformers/LaBSE", max_seq_length=None),
]
REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}
DEFAULT_MODELS = ["e5-small", "e5-base", "minilm-l12", "bge-m3"]
SMOKE_KEY = "smoke-hash"  # offline char-n-gram hashing encoder, used for tests/CI only


class SentenceTransformerEncoder:
    def __init__(self, spec: ModelSpec, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.spec = spec
        self.model = SentenceTransformer(spec.hf_id, device=device)
        if spec.max_seq_length:
            self.model.max_seq_length = spec.max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.device = str(self.model.device)
        self.num_params = int(sum(p.numel() for p in self.model.parameters()))

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        emb = self.model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix, batch_size)

    def encode_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, self.spec.doc_prefix, batch_size)


class HashingEncoder:
    """Dependency-free character n-gram hashing encoder (signed feature hashing)."""

    def __init__(self, dim: int = 1024, ngram: tuple[int, int] = (2, 4)):
        self.dim, self.ngram = dim, ngram
        self.device = "cpu"
        self.num_params = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {text} "
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                h = zlib.crc32(t[i : i + n].encode("utf-8"))
                v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts)

    def encode_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts)


def load_encoder(key: str, device: str | None = None):
    if key == SMOKE_KEY:
        return HashingEncoder()
    if key not in REGISTRY:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(REGISTRY)} or {SMOKE_KEY!r}")
    return SentenceTransformerEncoder(REGISTRY[key], device=device)


def hf_id(key: str) -> str:
    return REGISTRY[key].hf_id if key in REGISTRY else key
