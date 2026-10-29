"""Build the evaluation set: N queries, each with ONE gold document.

--mode llm   (recommended) Claude writes a natural Persian question that the document answers.
             Needs `pip install anthropic` and ANTHROPIC_API_KEY. Output is a *draft*:
             open the review CSV, delete or fix bad/ambiguous questions, then re-save.
--mode title (offline, weak) query = article title. Easy and lexically biased (the lead usually
             repeats the title) -> use only as a sanity baseline.

    python -m pss.build_eval --n-queries 100 --mode llm
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time

from .data import load_corpus, write_jsonl

PROMPT = """متن زیر ابتدای یک مقاله‌ی ویکی‌پدیای فارسی است.

عنوان: {title}
متن:
{text}

یک پرسش طبیعی و کوتاه به زبان فارسی بنویس که پاسخش در همین متن باشد، شبیه چیزی که یک کاربر واقعی در جست‌وجو می‌نویسد.
شرایط:
- پرسش باید آن‌قدر مشخص باشد که فقط به همین مقاله اشاره کند (از جزئیات متن استفاده کن، نه یک سؤال کلی).
- عنوان مقاله را عیناً در پرسش نیاور و جمله‌ای از متن را کلمه‌به‌کلمه کپی نکن.
- فقط خود پرسش را بنویس، بدون توضیح، بدون نقل‌قول."""


def llm_question(client, model: str, title: str, text: str) -> str:
    msg = client.messages.create(model=model, max_tokens=200,
                                 messages=[{"role": "user", "content": PROMPT.format(title=title, text=text)}])
    return msg.content[0].text.strip().strip('"«»').strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--mode", choices=["llm", "title"], default="llm")
    ap.add_argument("--llm-model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/queries.jsonl")
    ap.add_argument("--review-csv", default="data/queries_review.csv")
    a = ap.parse_args()

    docs = load_corpus(a.corpus)
    picked = random.Random(a.seed).sample(docs, min(a.n_queries, len(docs)))
    rows = []
    if a.mode == "llm":
        import anthropic  # lazy

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()
        for i, d in enumerate(picked, 1):
            for attempt in range(3):
                try:
                    q = llm_question(client, a.llm_model, d.title, d.text)
                    break
                except Exception as e:  # rate limit / transient errors
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
                    print(f"retry {i}: {e}")
            rows.append({"qid": f"q{i:03d}", "query": q, "gold_id": d.id, "gold_title": d.title})
            print(f"{i:>3}/{len(picked)}  {d.title}  ->  {q}")
    else:
        rows = [{"qid": f"q{i:03d}", "query": d.title, "gold_id": d.id, "gold_title": d.title}
                for i, d in enumerate(picked, 1)]

    write_jsonl(a.out, rows)
    with open(a.review_csv, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens correctly in Excel
        w = csv.writer(f)
        w.writerow(["qid", "query", "gold_title", "gold_id"])
        w.writerows([[r["qid"], r["query"], r["gold_title"], r["gold_id"]] for r in rows])
    print(f"\nwrote {len(rows)} queries to {a.out}\nREVIEW them: read {a.review_csv}, then delete the lines of ambiguous questions from {a.out} before benchmarking.")


if __name__ == "__main__":
    main()
