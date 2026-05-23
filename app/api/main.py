"""FastAPI service exposing /v1/analyze and health endpoints."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.orchestrator import run_analysis

configure_logging()
log = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Multimodal Financial & Legal Analyst",
    version="0.1.0",
    description="Ingests PDF/Excel/Image and emits a cited, guardrailed analyst report.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env, "model": settings.llm_model}


@app.post("/v1/analyze")
async def analyze(
    files: list[UploadFile] = File(..., description="PDF/XLSX/Image artifacts"),
    mode: Literal["investment", "legal"] = Form("investment"),
    topic: str = Form(""),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files supplied.")

    request_id = uuid.uuid4().hex[:12]
    tmp_root = Path(tempfile.mkdtemp(prefix=f"analyst_{request_id}_"))
    saved: list[Path] = []
    try:
        max_bytes = settings.max_file_mb * 1024 * 1024
        for f in files:
            data = await f.read()
            if len(data) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {f.filename!r} exceeds {settings.max_file_mb}MB limit.",
                )
            safe_name = Path(f.filename or "upload").name  # strip path traversal
            target = tmp_root / safe_name
            target.write_bytes(data)
            saved.append(target)

        log.info("api.analyze.start", request_id=request_id, files=[p.name for p in saved], mode=mode)
        result = run_analysis(saved, mode=mode, topic=topic)
        log.info(
            "api.analyze.done",
            request_id=request_id,
            issues=len(result.guardrails.issues),
            repaired=result.repaired,
        )

        return JSONResponse(
            {
                "request_id": request_id,
                "mode": result.mode,
                "report": result.report.model_dump(),
                "guardrails": result.guardrails.model_dump(),
                "queries": result.queries,
                "used_chunks": result.used_chunks,
                "repaired": result.repaired,
            }
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("api.analyze.error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
