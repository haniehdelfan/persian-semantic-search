"""Turn benchmark rows into markdown tables, a recommendation and a trade-off plot."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TOLERANCE = 0.02  # a model counts as "as good as the best" if its MRR@10 is within 0.02


def _get(rows, model, variant, qset):
    for r in rows:
        if r["model"] == model and r["variant"] == variant and r["query_set"] == qset:
            return r
    return None


def _models(rows) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def _table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _f(x, nd=3):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def recommend(rows) -> dict | None:
    """Smallest model whose MRR@10 (full normalization, clean queries) is within TOLERANCE of the best."""
    cands = [r for r in rows if r["variant"] == "full" and r["query_set"] == "clean" and r["model"] != "bm25"]
    if not cands:
        return None
    best = max(cands, key=lambda r: r["mrr@10"])
    ok = [r for r in cands if best["mrr@10"] - r["mrr@10"] <= TOLERANCE]
    pick = min(ok, key=lambda r: (r["params_m"], r["q_encode_p50_ms"]))
    return {"best": best, "pick": pick}


def make_markdown(rows: list[dict]) -> str:
    models = _models(rows)
    out: list[str] = []
    n_q = rows[0]["n_queries"] if rows else 0

    # 1) model comparison -------------------------------------------------------
    body = []
    for m in models:
        r = _get(rows, m, "full", "clean")
        if not r:
            continue
        body.append([
            m, _f(r["params_m"], 0), str(r["dim"]), _f(r["size_mb_fp32"], 0), _f(r["index_mb"], 1),
            _f(r["docs_per_s"], 1), _f(r["q_encode_p50_ms"], 1), _f(r["q_encode_p95_ms"], 1),
            f"{r['mrr@10']:.3f} [{r['mrr@10_lo']:.2f}–{r['mrr@10_hi']:.2f}]",
            _f(r["recall@1"], 2), _f(r["recall@5"], 2), _f(r["recall@10"], 2),
        ])
    out += [f"### Quality vs. speed vs. size  (normalization = full, clean queries, {n_q} queries)", "",
            _table(["Model", "Params (M)", "Dim", "Weights fp32 (MB)", "Index (MB)", "Docs/s (encode)",
                    "Query enc. p50 (ms)", "p95 (ms)", "MRR@10 [95% CI]", "R@1", "R@5", "R@10"], body), ""]

    # 2) normalization effect ----------------------------------------------------
    body = []
    for m in models:
        cells = [m]
        for qset in ("clean", "noisy"):
            a, b = _get(rows, m, "none", qset), _get(rows, m, "full", qset)
            if a and b:
                cells += [_f(a["mrr@10"]), _f(b["mrr@10"]), f"{b['mrr@10'] - a['mrr@10']:+.3f}"]
            else:
                cells += ["–", "–", "–"]
        body.append(cells)
    out += ["### Effect of normalization (MRR@10)", "",
            _table(["Model", "clean: none", "clean: full", "Δ", "noisy: none", "noisy: full", "Δ"], body), "",
            "`noisy` = queries typed with an Arabic keyboard (ي/ك), Arabic-Indic digits and no half-space.", ""]

    # 3) ablation ---------------------------------------------------------------------
    abl = [m for m in models if len({r["variant"] for r in rows if r["model"] == m}) > 2]
    for m in abl:
        base_c, base_n = _get(rows, m, "none", "clean"), _get(rows, m, "none", "noisy")
        body = []
        for v in [r["variant"] for r in rows if r["model"] == m and r["query_set"] == "clean"]:
            c, n = _get(rows, m, v, "clean"), _get(rows, m, v, "noisy")
            body.append([v, _f(c["mrr@10"]), f"{c['mrr@10'] - base_c['mrr@10']:+.3f}",
                         _f(n["mrr@10"]), f"{n['mrr@10'] - base_n['mrr@10']:+.3f}"])
        out += [f"### Normalization ablation — `{m}` (each step alone, vs. `none`)", "",
                _table(["Step", "clean MRR@10", "Δ", "noisy MRR@10", "Δ"], body), ""]

    # 4) recommendation ------------------------------------------------------------------
    rec = recommend(rows)
    if rec:
        b, p = rec["best"], rec["pick"]
        out += ["### Recommendation (auto-generated)", ""]
        if b["model"] == p["model"]:
            out.append(f"Best model: **{b['model']}** (MRR@10 = {b['mrr@10']:.3f}, query p50 = "
                       f"{b['q_encode_p50_ms']:.1f} ms, {b['params_m']:.0f}M params).")
        else:
            out.append(f"Best quality: **{b['model']}** (MRR@10 = {b['mrr@10']:.3f}, {b['params_m']:.0f}M params, "
                       f"p50 = {b['q_encode_p50_ms']:.1f} ms). Smallest model within {TOLERANCE} MRR of the best: "
                       f"**{p['model']}** (MRR@10 = {p['mrr@10']:.3f}, {p['params_m']:.0f}M params, "
                       f"p50 = {p['q_encode_p50_ms']:.1f} ms).")
        bm = _get(rows, "bm25", "full", "clean")
        if bm:
            out.append(f"\nLexical BM25 baseline: MRR@10 = {bm['mrr@10']:.3f}.")
        out.append("")
    return "\n".join(out)


def save_results(rows: list[dict], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if rows:
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = make_markdown(rows)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    plot_tradeoff(rows, out_dir / "tradeoff.png")


def update_readme(readme: str | Path, markdown: str) -> bool:
    p = Path(readme)
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8")
    pat = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), re.S)
    if not pat.search(text):
        return False
    p.write_text(pat.sub(lambda _: f"{MARK_START}\n{markdown}\n{MARK_END}", text), encoding="utf-8")
    return True


def plot_tradeoff(rows: list[dict], path: str | Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    pts = [r for r in rows if r["variant"] == "full" and r["query_set"] == "clean" and r["model"] != "bm25"]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r in pts:
        ax.scatter(r["q_encode_p50_ms"], r["mrr@10"], s=40 + r["params_m"], alpha=0.7)
        ax.annotate(r["model"], (r["q_encode_p50_ms"], r["mrr@10"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    bm = _get(rows, "bm25", "full", "clean")
    if bm:
        ax.axhline(bm["mrr@10"], ls="--", lw=1, color="gray")
        ax.annotate("BM25", (ax.get_xlim()[0], bm["mrr@10"]), textcoords="offset points", xytext=(4, 3), fontsize=8, color="gray")
    ax.set_xlabel("single-query encode latency, p50 (ms)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Quality vs. latency (bubble size = #params)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
