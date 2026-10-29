export PYTHONPATH := src
MODEL ?= e5-base

.PHONY: data eval-set eval index serve test smoke

data:
	python -m pss.prepare_data --n-docs 5000

eval-set:
	python -m pss.build_eval --n-queries 100 --mode llm

eval:
	python -m pss.run_eval --ablation-model e5-small

index:
	python -m pss.build_index --model $(MODEL)

serve:
	PSS_INDEX_DIR=artifacts/index/$(MODEL) uvicorn pss.api:app --port 8000

test:
	python -m pytest -q

# offline end-to-end check: tiny corpus, no model downloads, does not touch README
smoke:
	python -c "import sys; sys.path.insert(0,'tests'); import conftest; from pss.data import Doc, save_docs; save_docs('artifacts/smoke_corpus.jsonl', [Doc(str(i),t,x) for i,(t,x) in enumerate(conftest._DOCS)])"
	python -m pss.build_eval --corpus artifacts/smoke_corpus.jsonl --n-queries 8 --mode title --out artifacts/smoke_queries.jsonl --review-csv artifacts/smoke_review.csv
	python -m pss.run_eval --corpus artifacts/smoke_corpus.jsonl --queries artifacts/smoke_queries.jsonl --models smoke-hash --ablation-model smoke-hash --out artifacts/smoke_results --readme --cache-dir artifacts/smoke_cache
