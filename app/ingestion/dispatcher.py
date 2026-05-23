"""Dispatch on file extension; return typed Chunks."""
from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.excel_loader import load_excel
from app.ingestion.image_loader import load_image
from app.ingestion.pdf_loader import load_pdf
from app.schemas import Chunk

log = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}


class UnsupportedFileType(ValueError):
    pass


def ingest_file(path: str | Path) -> list[Chunk]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    if ext in EXCEL_EXTS:
        return load_excel(p)
    if ext in IMAGE_EXTS:
        return load_image(p)
    raise UnsupportedFileType(f"Unsupported file type: {ext}")


def ingest_files(paths: list[str | Path]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for p in paths:
        try:
            all_chunks.extend(ingest_file(p))
        except UnsupportedFileType as e:
            log.warning("ingest.skipped", path=str(p), reason=str(e))
    log.info("ingest.complete", total_chunks=len(all_chunks))
    return all_chunks
