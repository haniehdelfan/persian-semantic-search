"""Exact (brute-force) cosine search. Vectors are L2-normalised, so dot product == cosine."""
from __future__ import annotations

import numpy as np


def topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k indices and scores along the last axis, sorted by descending score."""
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(scores, part, -1), axis=-1)
    idx = np.take_along_axis(part, order, -1)
    return idx, np.take_along_axis(scores, idx, -1)
