"""Excel loader: every sheet becomes one or more table chunks (markdown rendered)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from app.core.logging import get_logger
from app.schemas import Chunk, ModalityType, SourceLocator

log = get_logger(__name__)

MAX_ROWS_PER_CHUNK = 60


def _hash(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return df.to_string(index=False)


def load_excel(path: str | Path) -> list[Chunk]:
    path = Path(path)
    chunks: list[Chunk] = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue

        n = len(df)
        for start in range(0, n, MAX_ROWS_PER_CHUNK):
            sub = df.iloc[start : start + MAX_ROWS_PER_CHUNK]
            end = start + len(sub)
            md = _df_to_markdown(sub)
            text = f"Sheet: {sheet} (rows {start + 1}-{end})\n\n{md}"
            cid = _hash(path.name, sheet, str(start))
            cell_range = f"A{start + 2}:{chr(ord('A') + min(len(sub.columns), 25) - 1)}{end + 1}"
            chunks.append(
                Chunk(
                    id=f"xlsx::{cid}",
                    modality=ModalityType.EXCEL_TABLE,
                    text=text,
                    locator=SourceLocator(
                        file_name=path.name,
                        sheet=sheet,
                        cell_range=cell_range,
                    ),
                )
            )

    log.info("excel.loaded", file=path.name, chunks=len(chunks))
    return chunks
