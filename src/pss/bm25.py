"""Small BM25 (Okapi) implementation used as a lexical baseline."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for term, c in Counter(toks).items():
                post[term].append((i, c))
        self.postings = {t: (np.array([p[0] for p in v]), np.array([p[1] for p in v], dtype=np.float64)) for t, v in post.items()}
        self.idf = {t: math.log(1 + (self.n - len(v[0]) + 0.5) / (len(v[0]) + 0.5)) for t, v in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            idx, tf = self.postings[t]
            denom = tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / self.avgdl)
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / denom
        return s
