"""Run the full benchmark and write results/ (json, csv, md, png).

    python -m pss.run_eval                              # 4 models + BM25, none vs full normalization
    python -m pss.run_eval --ablation-model e5-small    # + per-step normalization ablation for that model
    python -m pss.run_eval --models smoke-hash          # offline smoke test (no downloads)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .bench import run_benchmark
from .data import load_corpus, load_queries
from .models import DEFAULT_MODELS
from .report import make_markdown, save_results, update_readme


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--ablation-model", default="e5-small", help="model that gets the per-step normalization ablation")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--cache-dir", default="artifacts/emb_cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--readme", nargs="*", default=["README.md", "README.fa.md"],
                    help="README files whose RESULTS block gets refreshed (pass no value to skip)")
    a = ap.parse_args()

    docs, queries = load_corpus(a.corpus), load_queries(a.queries)
    ids = {d.id for d in docs}
    missing = [q.qid for q in queries if q.gold_id not in ids]
    if missing:
        raise SystemExit(f"{len(missing)} queries point to documents that are not in the corpus: {missing[:5]} ...")
    print(f"corpus: {len(docs)} docs | queries: {len(queries)} | models: {a.models}")

    rows = run_benchmark(docs, queries, a.models, ablation_model=a.ablation_model, with_bm25=not a.no_bm25,
                         device=a.device, batch_size=a.batch_size, cache_dir=Path(a.cache_dir))
    save_results(rows, a.out)
    md = make_markdown(rows)
    for path in filter(None, a.readme):
        if update_readme(path, md):
            print(f"README results block updated ({path})")
    print("\n" + md)


if __name__ == "__main__":
    main()
