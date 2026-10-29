"""End-to-end smoke test with the offline hashing encoder (no model downloads)."""
import numpy as np
from fastapi.testclient import TestClient

from pss.api import create_app
from pss.bench import run_benchmark
from pss.build_index import build_index
from pss.data import save_docs
from pss.models import SMOKE_KEY, HashingEncoder
from pss.normalize import ZWNJ, normalize
from pss.perturb import noisy_query
from pss.report import make_markdown, recommend


def _mrr(rows, model, variant, qset):
    return next(r["mrr@10"] for r in rows if r["model"] == model and r["variant"] == variant and r["query_set"] == qset)


def test_benchmark_runs_and_normalization_helps_on_noisy_queries(toy):
    docs, queries = toy
    rows = run_benchmark(docs, queries, [SMOKE_KEY], ablation_model=SMOKE_KEY, with_bm25=True, log=lambda *_: None)
    assert _mrr(rows, SMOKE_KEY, "full", "clean") >= 0.8
    # normalization never hurts on noisy queries and brings them back to the clean level
    assert _mrr(rows, SMOKE_KEY, "full", "noisy") >= _mrr(rows, SMOKE_KEY, "none", "noisy")
    assert _mrr(rows, SMOKE_KEY, "full", "noisy") >= _mrr(rows, SMOKE_KEY, "full", "clean") - 0.15
    assert _mrr(rows, "bm25", "full", "clean") > 0.5
    md = make_markdown(rows)
    assert "Quality vs. speed" in md and "ablation" in md
    assert recommend(rows)["pick"]["model"] == SMOKE_KEY


def test_noise_is_repaired_by_normalization(toy):
    """Letters, digits and punctuation noise is fully undone. Half-spaces are only restored where a
    safe rule exists (می/نمی prefix, ها/های/تر/ترین suffix), so compare with ZWNJ flattened to a space
    and check the rule-covered case separately."""
    _, queries = toy

    def flat(s):
        return normalize(s).replace(ZWNJ, " ")

    assert noisy_query(queries[0].query) != queries[0].query
    for q in queries:
        assert flat(noisy_query(q.query)) == flat(q.query), q.query
    # rule-covered half-space is restored exactly
    q = queries[2].query  # contains «می‌شود»
    assert normalize(noisy_query(q)) == normalize(q)
    # documented limitation: a half-space inside a compound ("جهان‌گستر") is NOT restored
    assert normalize(noisy_query("وب جهان‌گستر")) != normalize("وب جهان‌گستر")


def test_api_end_to_end(toy, tmp_path):
    docs, queries = toy
    corpus = tmp_path / "corpus.jsonl"
    save_docs(corpus, docs)
    idx = build_index(SMOKE_KEY, str(corpus), str(tmp_path / "idx"), encoder=HashingEncoder())
    assert np.load(idx / "embeddings.npy").shape == (len(docs), 1024)

    app = create_app(index_dir=str(idx), encoder=HashingEncoder())
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/info").json()["model_key"] == SMOKE_KEY
        q = queries[0]
        r = c.post("/search", json={"query": q.query, "k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["hits"]) == 3 and body["hits"][0]["id"] == q.gold_id
        assert body["hits"][0]["score"] >= body["hits"][1]["score"]
        # Arabic-keyboard query is normalised before encoding
        r2 = c.get("/search", params={"q": noisy_query(queries[1].query), "k": 1}).json()
        assert r2["hits"][0]["id"] == queries[1].gold_id
        assert c.post("/search", json={"query": "", "k": 3}).status_code == 422
        assert c.post("/search", json={"query": "تهران", "k": 0}).status_code == 422
