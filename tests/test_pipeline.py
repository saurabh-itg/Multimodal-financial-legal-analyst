import pandas as pd

from app.orchestrator import run_analysis


def test_end_to_end_investment(tmp_path, tmp_chroma):
    p = tmp_path / "f.xlsx"
    df = pd.DataFrame({"Year": [2022, 2023], "Revenue": [100, 120]})
    df.to_excel(p, index=False, sheet_name="Sheet1")

    res = run_analysis([p], mode="investment", topic="Should we invest in ACME?")
    assert res.report.company
    assert res.guardrails is not None
    assert res.queries
