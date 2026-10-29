"""Build a search index (embeddings + docs + meta) for one model.

    python -m pss.build_index --model e5-base --corpus data/corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import load_corpus, write_jsonl
from .models import hf_id, load_encoder
from .normalize import normalize, preset


def build_index(model_key: str, corpus_path: str, out_dir: str, norm: str = "full",
                device: str | None = None, batch_size: int = 32, encoder=None) -> Path:
    docs = load_corpus(corpus_path)
    cfg = preset(norm)
    enc = encoder or load_encoder(model_key, device)
    t0 = time.perf_counter()
    texts = [normalize(d.text, cfg) for d in docs]
    emb = enc.encode_docs(texts, batch_size=batch_size).astype(np.float32)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    write_jsonl(out / "docs.jsonl", ({"id": d.id, "title": d.title, "text": d.text} for d in docs))
    (out / "meta.json").write_text(json.dumps({
        "model_key": model_key, "hf_id": hf_id(model_key), "norm": norm, "norm_config": asdict(cfg),
        "n_docs": len(docs), "dim": int(emb.shape[1]), "build_seconds": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="e5-base", help="registry key (e5-small, e5-base, minilm-l12, bge-m3, ...)")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default=None, help="default: artifacts/index/<model>")
    ap.add_argument("--norm", default="full", help="normalization preset (default: full)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    out = build_index(a.model, a.corpus, a.out or f"artifacts/index/{a.model}", a.norm, a.device, a.batch_size)
    print(f"index written to {out}")


if __name__ == "__main__":
    main()
