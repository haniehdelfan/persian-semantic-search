# Persian Semantic Search: Embedding Model Benchmark

[![CI](https://github.com/haniehdelfan/persian-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/haniehdelfan/persian-semantic-search/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md) | [فارسی](README.fa.md)

A reproducible benchmark of multilingual embedding models on **Persian** text, plus a small **FastAPI** search service built on the winner. The goal is to answer one practical question with data rather than reputation:

> *For Persian retrieval, which embedding model should I ship, and why?*

## Highlights

- **Four multilingual models compared** (`multilingual-e5-small`, `multilingual-e5-base`, `paraphrase-multilingual-MiniLM-L12-v2`, `bge-m3`) against a **BM25** lexical baseline, on a random sample of Persian Wikipedia.
- **Quality, speed and size together:** MRR@10 (with bootstrap 95% CIs), recall@1/5/10, nDCG@10, single-query p50/p95 latency, encode throughput, parameter count and index size.
- **Persian normalization measured in isolation:** each step (letter unification, digits, diacritics, punctuation, ZWNJ/half-space repair) is ablated on its own, on both clean queries and *noisy* queries that simulate real typing (Arabic keyboard, Arabic-Indic digits, missing half-space).
- **Evaluation set built with an LLM, then human-reviewed:** natural Persian questions with one gold document each.
- **Production-style serving:** FastAPI endpoint that applies the *same* normalization the index was built with.
- **Testable offline:** an end-to-end test suite runs with a dependency-free hashing encoder, so CI needs no model downloads.

## Architecture

```
 Persian Wikipedia ──► prepare_data ──► data/corpus.jsonl      (≈5k article leads)
                                              │
 build_eval (Claude + manual review) ──► data/queries.jsonl    (≈100 questions, 1 gold doc each)
                                              │
        ┌─────────────────────────────────────┴───────────────┐
        ▼                                                     ▼
   run_eval                                            build_index ──► artifacts/index/<model>/
   4 embedding models + BM25                                                      │
   × normalization (none / full / each step)                                      ▼
   × clean / noisy queries                                              FastAPI  POST /search
        │
        ▼
   results/ (json, csv, md, png) ──► results block in this README
```

## Quickstart

```bash
pip install -e ".[ml,dev]"

make data         # sample 5,000 Persian Wikipedia article leads (HF: wikimedia/wikipedia, 20231101.fa)
make eval-set     # generate ~100 questions with Claude (needs ANTHROPIC_API_KEY)
                  # → review: delete ambiguous lines from data/queries.jsonl (queries_review.csv is a read-only view)
make eval         # run the full benchmark; refreshes the results block below
make index MODEL=e5-base
make serve MODEL=e5-base
make test
```

No downloads needed for a quick end-to-end check: `make smoke`. To use your own corpus (e.g. Persian news), pass `--input-jsonl` to `pss.prepare_data` (`{"id","title","text"}` per line).

### Run it on GitHub Actions (no local setup)

Everything can run on GitHub's free CPU runners (public repositories get 4 vCPU / 16 GB; a job is limited to 6 hours):

1. Add the repository secret `ANTHROPIC_API_KEY` (Settings → Secrets and variables → Actions).
2. Actions → **Build evaluation set** → Run workflow. It commits draft questions to `data/`.
3. Review them in the browser: open `data/queries_review.csv` (rendered as a table), then edit `data/queries.jsonl` and delete the lines of ambiguous questions.
4. Actions → **Run benchmark** → Run workflow. It commits `results/` and fills the results block of this README.

Latency numbers from this setup come from a shared CPU runner, so treat them as relative comparisons between models, not absolute production figures.

## Evaluation design

| Aspect | Choice |
|---|---|
| Corpus | Random sample (reservoir sampling) of article leads, ≤1200 chars; lists and disambiguation pages removed |
| Queries | One natural question per sampled document, written by Claude without copying the title or a verbatim sentence; **manual review required** |
| Relevance | Exactly one gold document per query |
| Quality metrics | MRR@10, recall@1/5/10, nDCG@10; 95% bootstrap CI on MRR |
| Speed | Batch-size-1 query encode latency (p50/p95, after warm-up), search latency, docs/s at encode time |
| Size | Parameters, fp32 weight size, index size |
| Baseline | BM25 (Okapi) over the same normalized text |
| E5 prefixes | `query: ` / `passage: ` applied automatically |
| Max length | 512 tokens, except MiniLM which keeps its native 128 |

### Persian normalization

| Step | What it does |
|---|---|
| `chars` | NFKC; ي→ی, ك→ک, ة→ه, ۀ→ه, أ/إ→ا |
| `diacritics` | strip harakat and tatweel |
| `digits` | Persian and Arabic-Indic digits → ASCII |
| `punct` | ؟ ، ؛ « » → ASCII equivalents |
| `zwnj_fix` | canonical half-space: `می شود`→`می‌شود`, `کتاب ها`→`کتاب‌ها`, stray/duplicate ZWNJ removed |
| `zwnj_space` | replace every ZWNJ with a space (compared against `zwnj_fix`) |
| `whitespace` | collapse whitespace |

Wikipedia is already clean, so normalization is expected to do little on clean queries. The `noisy` query set rewrites the same questions the way users actually type them, which is where normalization should matter. Half-space restoration only uses safe rules (می/نمی prefix, ها/های/تر/ترین suffix); a half-space inside a compound such as «جهان‌گستر» is not recovered if the user omits it (covered by a test).

## Results

<!-- RESULTS:START -->
*Not generated yet. Run `make data && make eval-set && make eval` and this block is filled automatically with the model comparison, normalization effect, ablation and an auto-generated recommendation.*
<!-- RESULTS:END -->

Quality vs. latency plot (generated by `make eval`): `results/tradeoff.png`

### Model choice

Selection rule used by the report: **the smallest model whose MRR@10 is within 0.02 of the best one.**

> After running the benchmark, summarize your conclusion here in 3 to 5 lines: chosen model, MRR@10 / recall@5 / latency, whether the gap to the best model is within the confidence interval, and which normalization steps mattered.

## API

```bash
PSS_INDEX_DIR=artifacts/index/e5-base uvicorn pss.api:app --port 8000

curl -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "پایتخت ایران کدام شهر است؟", "k": 3}'
```

| Endpoint | Description |
|---|---|
| `POST /search` | body `{"query": str, "k": 1..50}` |
| `GET /search?q=...&k=5` | same, via query string |
| `GET /info` | model, normalization config, index size |
| `GET /health` | liveness |

The response contains `normalized_query`, `model`, `took_ms` and ranked `hits` (`id`, `title`, `score`, `snippet`). Search is exact brute-force cosine, which is fine for thousands of documents; for millions, move to an ANN index (FAISS, HNSW).

Docker (CPU): `docker build -t persian-semantic-search .`, build the index first, then mount `artifacts/` (see comments in the `Dockerfile`).

## Project layout

```
src/pss/
  normalize.py     Persian normalization, one flag per step
  perturb.py       realistic query noise (Arabic keyboard, digits, missing half-space)
  models.py        model registry, E5 prefixes, offline hashing encoder for tests
  metrics.py       MRR, recall@k, nDCG, bootstrap CI
  bm25.py          lexical baseline
  bench.py         benchmark runner (quality, latency, size) with embedding cache
  report.py        markdown tables, plot, auto-recommendation
  prepare_data.py  corpus builder
  build_eval.py    evaluation-set builder (LLM + review CSV)
  run_eval.py      full comparison CLI
  build_index.py   index builder
  api.py           FastAPI service
tests/             normalization, metrics, end-to-end pipeline and API (offline)
```

## Limitations

- Questions are LLM-generated, so their style differs from real user queries; manual review reduces but does not remove that bias.
- One gold document per query: if another document also answers it, every model is penalized equally.
- Only article leads are indexed, not full articles.
- Latencies depend on hardware (device is recorded in `results/results.csv`); compare them only within one run.
- With ~100 queries the confidence intervals are wide; differences smaller than the CI are not evidence of a better model.

## Ideas for extension

Hybrid retrieval (BM25 + dense, reciprocal rank fusion), a cross-encoder reranker, a public Persian retrieval benchmark (e.g. MIRACL-fa) as a second evaluation set, quantized/ONNX encoders for lower latency, and chunking for full-length articles.

## License

MIT, see [LICENSE](LICENSE).
