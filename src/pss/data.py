from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    query: str
    gold_id: str


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_corpus(path: str | Path) -> list[Doc]:
    return [Doc(id=str(r["id"]), title=r.get("title", ""), text=r["text"]) for r in read_jsonl(path)]


def load_queries(path: str | Path) -> list[Query]:
    return [Query(qid=str(r["qid"]), query=r["query"], gold_id=str(r["gold_id"])) for r in read_jsonl(path)]


def save_docs(path: str | Path, docs: Iterable[Doc]) -> int:
    return write_jsonl(path, (asdict(d) for d in docs))


def make_lead(text: str, max_chars: int = 1200) -> str:
    """First paragraphs of an article, cut at a sentence boundary when possible."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out = ""
    for p in paras:
        if len(out) + len(p) + 1 > max_chars:
            if not out:  # first paragraph alone is too long -> cut it
                cut = p[:max_chars]
                m = max(cut.rfind("."), cut.rfind("؟"), cut.rfind("!"))
                out = cut[: m + 1] if m > max_chars // 2 else cut
            break
        out = f"{out}\n{p}" if out else p
    return out.strip()


def corpus_fingerprint(docs: list[Doc]) -> str:
    h = hashlib.sha1()
    for d in docs:
        h.update(d.id.encode())
        h.update(d.text.encode())
    return h.hexdigest()[:12]
