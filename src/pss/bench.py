"""Benchmark runner: quality (MRR / recall@k / nDCG), speed and size for each model,
plus the normalization effect on clean and noisy queries."""
from __future__ import annotations

import hashlib
import pickle
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .data import Doc, Query
from .metrics import bootstrap_ci, ranks_from_topk, reciprocal_ranks, summarize
from .models import hf_id, load_encoder
from .normalize import normalize, preset
from .perturb import noisy_query
from .search import topk

TOPK = 10
MAIN_VARIANTS = ["none", "full"]
ABLATION_VARIANTS = ["none", "chars", "diacritics", "digits", "punct", "zwnj_fix", "zwnj_space", "whitespace", "full"]


class EmbeddingStore:
    """Text-hash -> vector cache (optionally persisted). Normalisation variants that leave a
    text unchanged reuse its embedding, so the ablation costs almost nothing on clean corpora."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        self.meta: dict = {}
        if path and path.exists():
            with open(path, "rb") as f:
                saved = pickle.load(f)
            self.vecs, self.meta = saved["vecs"], saved["meta"]

    @staticmethod
    def _key(role: str, text: str) -> str:
        return role + ":" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, role: str, texts: list[str], encode: Callable[[list[str]], np.ndarray]) -> tuple[np.ndarray, int]:
        keys = [self._key(role, t) for t in texts]
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self.vecs and k not in missing:
                missing[k] = t
        if missing:
            ks = list(missing)
            for k, v in zip(ks, encode([missing[k] for k in ks])):
                self.vecs[k] = v
        return np.stack([self.vecs[k] for k in keys]), len(missing)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                pickle.dump({"vecs": self.vecs, "meta": self.meta}, f)


def _row(model: str, variant: str, qset: str, ranks: np.ndarray) -> dict:
    s = summarize(ranks)
    lo, hi = bootstrap_ci(reciprocal_ranks(ranks, TOPK))
    return {"model": model, "variant": variant, "query_set": qset, "n_queries": int(len(ranks)),
            **s, "mrr@10_lo": lo, "mrr@10_hi": hi}


def _query_sets(queries: list[Query]) -> dict[str, list[str]]:
    return {"clean": [q.query for q in queries], "noisy": [noisy_query(q.query) for q in queries]}


def _pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def bench_model(key: str, docs: list[Doc], queries: list[Query], variants: list[str], *,
                device: str | None = None, batch_size: int = 32, cache_dir: Path | None = None,
                log=print) -> list[dict]:
    log(f"[{key}] loading model ...")
    enc = load_encoder(key, device)
    store = EmbeddingStore(cache_dir / f"{key}.pkl" if cache_dir else None)
    doc_index = {d.id: i for i, d in enumerate(docs)}
    gold = np.array([doc_index[q.gold_id] for q in queries])
    qsets = _query_sets(queries)
    rows: list[dict] = []
    D_by_variant: dict[str, np.ndarray] = {}

    for vname in variants:
        cfg = preset(vname)
        dtexts = [normalize(d.text, cfg) for d in docs]
        t0 = time.perf_counter()
        D, n_new = store.get("doc", dtexts, lambda x: enc.encode_docs(x, batch_size))
        dt = time.perf_counter() - t0
        if "docs_per_s" not in store.meta and n_new >= 0.9 * len(set(dtexts)):
            store.meta["docs_per_s"] = n_new / dt  # only a (near-)full encode gives a fair throughput
        D_by_variant[vname] = D
        for qset, texts in qsets.items():
            qn = [normalize(t, cfg) for t in texts]
            Q, _ = store.get("query", qn, lambda x: enc.encode_queries(x, batch_size))
            idx, _ = topk(Q @ D.T, TOPK)
            rows.append(_row(key, vname, qset, ranks_from_topk(idx, gold)))
        log(f"[{key}] variant={vname:<11} " + " ".join(
            f"{r['query_set']}:MRR@10={r['mrr@10']:.3f}" for r in rows[-2:]))

    # --- latency: single-query, batch size 1, no cache ------------------------------
    lat_variant = "full" if "full" in D_by_variant else variants[-1]
    D = D_by_variant[lat_variant]
    cfg = preset(lat_variant)
    qn = [normalize(q.query, cfg) for q in queries]
    for q in qn[:3]:
        enc.encode_queries([q])  # warm-up
    enc_ms, search_ms = [], []
    for q in qn:
        t0 = time.perf_counter()
        v = enc.encode_queries([q])
        t1 = time.perf_counter()
        topk(D @ v[0], TOPK)
        t2 = time.perf_counter()
        enc_ms.append((t1 - t0) * 1e3)
        search_ms.append((t2 - t1) * 1e3)
    info = {
        "hf_id": hf_id(key), "dim": enc.dim, "device": enc.device,
        "params_m": enc.num_params / 1e6, "size_mb_fp32": enc.num_params * 4 / 2**20,
        "index_mb": len(docs) * enc.dim * 4 / 2**20,
        "docs_per_s": store.meta.get("docs_per_s", float("nan")),
        "q_encode_p50_ms": _pct(enc_ms, 50), "q_encode_p95_ms": _pct(enc_ms, 95),
        "search_p50_ms": _pct(search_ms, 50),
    }
    store.save()
    return [{**r, **info} for r in rows]


def bench_bm25(docs: list[Doc], queries: list[Query], variants: list[str], log=print) -> list[dict]:
    doc_index = {d.id: i for i, d in enumerate(docs)}
    gold = np.array([doc_index[q.gold_id] for q in queries])
    qsets = _query_sets(queries)
    rows, lat = [], []
    for vname in variants:
        cfg = preset(vname)
        bm = BM25([tokenize(normalize(d.text, cfg)) for d in docs])
        for qset, texts in qsets.items():
            idx_rows = []
            for t in texts:
                t0 = time.perf_counter()
                s = bm.scores(tokenize(normalize(t, cfg)))
                idx_rows.append(topk(s, TOPK)[0])
                if vname == variants[-1] and qset == "clean":
                    lat.append((time.perf_counter() - t0) * 1e3)
            rows.append(_row("bm25", vname, qset, ranks_from_topk(np.stack(idx_rows), gold)))
        log(f"[bm25] variant={vname:<11} " + " ".join(
            f"{r['query_set']}:MRR@10={r['mrr@10']:.3f}" for r in rows[-2:]))
    info = {"hf_id": "(lexical baseline)", "dim": 0, "device": "cpu", "params_m": 0.0, "size_mb_fp32": 0.0,
            "index_mb": 0.0, "docs_per_s": float("nan"), "q_encode_p50_ms": 0.0, "q_encode_p95_ms": 0.0,
            "search_p50_ms": _pct(lat, 50)}
    return [{**r, **info} for r in rows]


def run_benchmark(docs, queries, models: list[str], *, ablation_model: str | None = None,
                  variants: list[str] | None = None, with_bm25: bool = True, **kw) -> list[dict]:
    variants = variants or MAIN_VARIANTS
    rows: list[dict] = []
    for key in models:
        v = ABLATION_VARIANTS if key == ablation_model else variants
        rows += bench_model(key, docs, queries, v, **kw)
    if with_bm25:
        rows += bench_bm25(docs, queries, ABLATION_VARIANTS if ablation_model == "bm25" else variants)
    return rows
