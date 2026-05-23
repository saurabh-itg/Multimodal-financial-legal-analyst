"""PDF loader: extracts text per page (with bbox) and embedded figures (rendered + described)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from app.core.logging import get_logger
from app.llm import describe_image
from app.schemas import Chunk, ModalityType, SourceLocator

log = get_logger(__name__)

_FIGURE_PROMPT = (
    "Describe this figure as if for a financial or legal analyst. "
    "Include type (chart/diagram/photo/text), axes, units, series, headline numbers, "
    "labels, and any visible captions. Be concise but exhaustive."
)


def _hash(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


def load_pdf(path: str | Path, *, work_dir: Path | None = None) -> list[Chunk]:
    """Return text chunks per text-block and figure chunks per embedded image."""
    path = Path(path)
    work_dir = work_dir or path.parent / ".figures"
    work_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    doc = fitz.open(path)
    try:
        for page_idx, page in enumerate(doc, start=1):
            # --- text blocks (give us bboxes) ---
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = block
                text = (text or "").strip()
                if len(text) < 20:  # ignore boilerplate
                    continue
                cid = _hash(path.name, str(page_idx), str(x0), str(y0), text[:40])
                chunks.append(
                    Chunk(
                        id=f"pdf::{cid}",
                        modality=ModalityType.PDF_TEXT,
                        text=text,
                        locator=SourceLocator(
                            file_name=path.name,
                            page=page_idx,
                            bbox=(float(x0), float(y0), float(x1), float(y1)),
                        ),
                    )
                )

            # --- embedded figures ---
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                except Exception as e:  # noqa: BLE001
                    log.warning("pdf.figure.extract_failed", error=str(e), page=page_idx)
                    continue

                # Skip tiny decorative icons
                try:
                    pil = Image.open(io.BytesIO(img_bytes))
                    if pil.width < 80 or pil.height < 80:
                        continue
                except Exception:  # noqa: BLE001
                    continue

                fig_id = _hash(path.name, str(page_idx), str(img_idx), str(xref))
                fig_path = work_dir / f"{path.stem}_p{page_idx}_f{img_idx}_{fig_id}.png"
                fig_path.write_bytes(img_bytes)

                try:
                    description = describe_image(fig_path, _FIGURE_PROMPT)
                except Exception as e:  # noqa: BLE001
                    log.warning("pdf.figure.describe_failed", error=str(e))
                    description = "[Figure could not be described]"

                chunks.append(
                    Chunk(
                        id=f"pdffig::{fig_id}",
                        modality=ModalityType.PDF_FIGURE,
                        text=description,
                        locator=SourceLocator(
                            file_name=path.name,
                            page=page_idx,
                            image_id=str(fig_path.name),
                        ),
                        extra={"image_path": str(fig_path)},
                    )
                )
    finally:
        doc.close()

    log.info("pdf.loaded", file=path.name, chunks=len(chunks))
    return chunks
