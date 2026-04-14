from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from services.indexer import get_index
from services.models import CompareResponse, RunDetail, RunListItem

app = FastAPI(title="CoT-BERT Experiment Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "EXPERIMENT_VIZ_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    idx = get_index()
    return {
        "status": "ok",
        "data_dirs": [str(p) for p in idx.data_dirs],
        "n_runs": len(get_index()),
    }


@app.post("/api/refresh")
def refresh_index() -> dict:
    idx = get_index()
    idx.refresh()
    return {"status": "refreshed", "n_runs": len(idx)}


@app.get("/api/runs", response_model=List[RunListItem])
def list_runs(
    q: Optional[str] = Query(None, description="Filter by run_id / tag substring"),
    tab: str = Query(
        "all",
        description="all = 全部 run 每行一个 run_id；senteval | alignment | gold 为子集过滤",
    ),
    limit: int = Query(200, ge=1, le=2000),
) -> List[RunListItem]:
    t = (tab or "all").strip().lower()
    if t not in ("all", "senteval", "alignment", "gold"):
        raise HTTPException(
            status_code=400,
            detail="tab must be all | senteval | alignment | gold",
        )
    idx = get_index()
    items = idx.list_runs(tab=t)
    if q and q.strip():
        ql = q.strip().lower()
        items = [
            x
            for x in items
            if ql in x.run_id.lower()
            or (x.eval_run_tag and ql in x.eval_run_tag.lower())
        ]
    return items[:limit]


@app.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    d = get_index().get_detail(run_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return d


@app.get("/api/compare", response_model=CompareResponse)
def compare(run_ids: str = Query(..., description="Comma-separated run ids")):
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if len(ids) < 1:
        raise HTTPException(status_code=400, detail="Need at least one run_id")
    if len(ids) > 12:
        raise HTTPException(status_code=400, detail="Too many runs (max 12)")
    data = get_index().compare(ids)
    if not data:
        raise HTTPException(status_code=404, detail="No matching runs")
    return CompareResponse(**data)


@app.get("/api/runs/{run_id}/files/{filename}")
def get_run_file(run_id: str, filename: str):
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    detail = get_index().get_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    for art in detail.artifacts:
        if art.name == filename:
            path = Path(art.path)
            if path.is_file():
                return FileResponse(
                    path,
                    filename=filename,
                    media_type=_guess_media(filename),
                )
    raise HTTPException(status_code=404, detail="File not found for this run")


def _guess_media(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


# 生产环境：npm run build 后将 frontend/dist/* 拷入 backend/static/，即可同端口提供前端（/api 仍优先）
_static = Path(__file__).resolve().parent / "static"
if _static.is_dir() and (_static / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
