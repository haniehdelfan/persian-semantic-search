"""Build the corpus (a few thousand Persian documents).

Default source: Persian Wikipedia from Hugging Face (`wikimedia/wikipedia`, config `20231101.fa`),
streamed and reservoir-sampled so that the corpus is a random sample, not "articles starting with آ".
Each document is the *lead* of an article (first paragraphs, <= --max-chars) so it fits the
512-token limit of the embedding models.

    python -m pss.prepare_data --n-docs 5000
    python -m pss.prepare_data --input-jsonl my_news.jsonl   # own corpus: {"id","title","text"} per line
"""
from __future__ import annotations

import argparse
import random

from .data import Doc, make_lead, read_jsonl, save_docs

_SKIP_TITLE = ("فهرست", "ویکی‌پدیا:", "رده:")
_DISAMBIG = "ممکن است به این موارد اشاره داشته باشد"


def _eligible(title: str, text: str, min_chars: int) -> bool:
    return len(text) >= min_chars and not title.startswith(_SKIP_TITLE) and _DISAMBIG not in text[:300]


def sample_wikipedia(n_docs: int, scan_limit: int, min_chars: int, max_chars: int, seed: int) -> list[Doc]:
    from datasets import load_dataset  # lazy

    ds = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
    rng = random.Random(seed)
    reservoir: list[Doc] = []
    seen = 0
    for row in ds:
        lead = make_lead(row["text"], max_chars)
        if not _eligible(row["title"], lead, min_chars):
            continue
        seen += 1
        doc = Doc(id=str(row["id"]), title=row["title"], text=lead)
        if len(reservoir) < n_docs:
            reservoir.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n_docs:
                reservoir[j] = doc
        if seen >= scan_limit:
            break
    return reservoir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--scan-limit", type=int, default=100_000, help="eligible articles to scan before stopping")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-jsonl", default=None, help="use your own corpus instead of Wikipedia")
    ap.add_argument("--out", default="data/corpus.jsonl")
    a = ap.parse_args()

    if a.input_jsonl:
        docs = [Doc(str(r["id"]), r.get("title", ""), make_lead(r["text"], a.max_chars)) for r in read_jsonl(a.input_jsonl)]
        docs = [d for d in docs if len(d.text) >= a.min_chars]
        random.Random(a.seed).shuffle(docs)
        docs = docs[: a.n_docs]
    else:
        docs = sample_wikipedia(a.n_docs, a.scan_limit, a.min_chars, a.max_chars, a.seed)
    n = save_docs(a.out, docs)
    print(f"wrote {n} documents to {a.out}")


if __name__ == "__main__":
    main()
