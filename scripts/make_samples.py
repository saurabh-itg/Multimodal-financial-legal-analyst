"""Generate a tiny sample Excel + dummy chart image for local smoke testing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Excel
    df = pd.DataFrame(
        {
            "Year": [2021, 2022, 2023],
            "Revenue ($M)": [820, 1010, 1230],
            "Net Income ($M)": [60, 95, 140],
            "Gross Margin %": [42.0, 44.5, 46.2],
        }
    )
    df.to_excel(out / "financials.xlsx", index=False, sheet_name="Annual")

    # Image (dummy "chart")
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((20, 20), "Revenue ($M) by Year", fill="black", font=font)
    bars = [(140, 280, 200, 250), (260, 280, 320, 200), (380, 280, 440, 140)]
    for x0, y0, x1, y1 in bars:
        d.rectangle([x0, y1, x1, y0], outline="black", fill="#4a90e2")
    d.text((150, 300), "2021  820", fill="black", font=font)
    d.text((270, 300), "2022 1010", fill="black", font=font)
    d.text((390, 300), "2023 1230", fill="black", font=font)
    img.save(out / "revenue_chart.png")

    print(f"Wrote samples to {out.resolve()}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "samples")
