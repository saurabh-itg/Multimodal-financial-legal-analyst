"""Standalone image loader: vision model produces a structured description chunk."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.logging import get_logger
from app.llm import describe_image
from app.schemas import Chunk, ModalityType, SourceLocator

log = get_logger(__name__)

_PROMPT = (
    "You are an analyst studying a chart, table screenshot, diagram, or scanned document. "
    "Describe its content, axes, units, series, headline numbers, labels, and any text you "
    "can read. Be specific. Do not speculate."
)


def load_image(path: str | Path) -> list[Chunk]:
    path = Path(path)
    description = describe_image(path, _PROMPT)
    cid = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:16]
    chunk = Chunk(
        id=f"img::{cid}",
        modality=ModalityType.IMAGE,
        text=description,
        locator=SourceLocator(file_name=path.name, image_id=path.name),
        extra={"image_path": str(path)},
    )
    log.info("image.loaded", file=path.name, chars=len(description))
    return [chunk]
