"""Retrieval metrics for the single-relevant-document setting (one gold doc per query)."""
from __future__ import annotations

import numpy as np

DEFAULT_KS = (1, 5, 10)


def ranks_from_topk(topk_idx: np.ndarray, gold_idx: np.ndarray) -> np.ndarray:
    """1-based rank of the gold doc inside each query's top-k list; 0 if it is absent."""
    hits = topk_idx == np.asarray(gold_idx)[:, None]
    found = hits.any(axis=1)
    return np.where(found, hits.argmax(axis=1) + 1, 0)


def reciprocal_ranks(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    ok = (ranks > 0) & (ranks <= k)
    return np.where(ok, 1.0 / np.maximum(ranks, 1), 0.0)


def summarize(ranks: np.ndarray, ks=DEFAULT_KS, mrr_k: int = 10) -> dict[str, float]:
    ranks = np.asarray(ranks)
    out = {f"mrr@{mrr_k}": float(reciprocal_ranks(ranks, mrr_k).mean())}
    for k in ks:
        out[f"recall@{k}"] = float(((ranks > 0) & (ranks <= k)).mean())
    ok = (ranks > 0) & (ranks <= mrr_k)
    out[f"ndcg@{mrr_k}"] = float(np.where(ok, 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0).mean())
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (per-query values, e.g. reciprocal ranks)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
