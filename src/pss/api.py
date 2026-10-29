"""FastAPI search service.

    PSS_INDEX_DIR=artifacts/index/e5-base uvicorn pss.api:app --port 8000
    curl -X POST localhost:8000/search -H 'content-type: application/json' \
         -d '{"query": "پایتخت ایران کدام شهر است؟", "k": 5}'

(No `from __future__ import annotations` here on purpose: FastAPI/Pydantic resolve the
annotations of these models at import time.)
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .data import read_jsonl
from .models import load_encoder
from .normalize import NormConfig, normalize
from .search import topk


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(10, ge=1, le=50)


class Hit(BaseModel):
    rank: int
    id: str
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    model: str
    took_ms: float
    hits: list[Hit]


def create_app(index_dir: str | None = None, encoder=None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        d = Path(index_dir or os.environ.get("PSS_INDEX_DIR", "artifacts/index/e5-base"))
        if not (d / "meta.json").exists():
            raise RuntimeError(f"no index found in {d}; run `python -m pss.build_index` first (or set PSS_INDEX_DIR)")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        state.update(
            meta=meta,
            emb=np.load(d / "embeddings.npy"),
            docs=list(read_jsonl(d / "docs.jsonl")),
            cfg=NormConfig(**meta["norm_config"]),
            enc=encoder or load_encoder(meta["model_key"], os.environ.get("PSS_DEVICE")),
        )
        yield
        state.clear()

    app = FastAPI(title="Persian semantic search", version="0.1.0", lifespan=lifespan)

    def _search(query: str, k: int) -> SearchResponse:
        t0 = time.perf_counter()
        qn = normalize(query, state["cfg"])
        if not qn:
            raise HTTPException(status_code=422, detail="query is empty after normalization")
        v = state["enc"].encode_queries([qn])[0]
        idx, scores = topk(state["emb"] @ v, k)
        hits = []
        for rank, (i, s) in enumerate(zip(idx, scores), start=1):
            d = state["docs"][int(i)]
            hits.append(Hit(rank=rank, id=str(d["id"]), title=d.get("title", ""), score=float(s),
                            snippet=d["text"][:200]))
        return SearchResponse(query=query, normalized_query=qn, model=state["meta"]["model_key"],
                              took_ms=round((time.perf_counter() - t0) * 1e3, 2), hits=hits)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return state["meta"]

    @app.post("/search", response_model=SearchResponse)
    def search_post(req: SearchRequest):
        return _search(req.query, req.k)

    @app.get("/search", response_model=SearchResponse)
    def search_get(q: str = Query(..., min_length=1, max_length=1000), k: int = Query(10, ge=1, le=50)):
        return _search(q, k)

    return app


app = create_app()
