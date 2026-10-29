import numpy as np

from pss.metrics import bootstrap_ci, ranks_from_topk, summarize
from pss.search import topk


def test_topk_sorted_and_correct():
    s = np.array([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.8, 0.1]])
    idx, sc = topk(s, 2)
    assert idx.tolist() == [[1, 3], [2, 0]]
    assert np.allclose(sc, [[0.9, 0.7], [0.8, 0.3]])
    assert topk(np.array([0.2, 0.8, 0.5]), 10)[0].tolist() == [1, 2, 0]  # k > n, 1-D input


def test_ranks_and_summary():
    top = np.array([[5, 1, 2], [4, 5, 6], [7, 8, 9]])
    gold = np.array([5, 6, 0])  # rank 1, rank 3, missing
    ranks = ranks_from_topk(top, gold)
    assert ranks.tolist() == [1, 3, 0]
    s = summarize(ranks, ks=(1, 3), mrr_k=10)
    assert np.isclose(s["mrr@10"], (1 + 1 / 3 + 0) / 3)
    assert np.isclose(s["recall@1"], 1 / 3)
    assert np.isclose(s["recall@3"], 2 / 3)
    assert np.isclose(s["ndcg@10"], (1 + 1 / np.log2(4)) / 3)


def test_mrr_cutoff():
    assert summarize(np.array([11]), mrr_k=10)["mrr@10"] == 0.0


def test_bootstrap_ci_contains_mean():
    v = np.array([1.0, 0.5, 0.0, 1.0, 0.333, 0.25, 1.0, 0.0])
    lo, hi = bootstrap_ci(v)
    assert lo <= v.mean() <= hi
