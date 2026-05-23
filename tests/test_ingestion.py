import pandas as pd

from app.ingestion import ingest_file
from app.schemas import ModalityType


def test_excel_loader(tmp_path):
    p = tmp_path / "f.xlsx"
    df = pd.DataFrame({"Year": [2022, 2023], "Revenue": [100, 120]})
    df.to_excel(p, index=False, sheet_name="Sheet1")
    chunks = ingest_file(p)
    assert chunks
    assert chunks[0].modality == ModalityType.EXCEL_TABLE
    assert "Revenue" in chunks[0].text
    assert chunks[0].locator.sheet == "Sheet1"


def test_image_loader(tmp_path):
    from PIL import Image

    p = tmp_path / "chart.png"
    Image.new("RGB", (200, 200), color=(255, 0, 0)).save(p)
    chunks = ingest_file(p)
    assert len(chunks) == 1
    assert chunks[0].modality == ModalityType.IMAGE
    assert chunks[0].locator.image_id == "chart.png"
