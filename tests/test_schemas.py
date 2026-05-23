from app.schemas import (
    Citation,
    Claim,
    InvestmentThesis,
    LegalRisk,
    LegalRiskReport,
)


def test_claim_requires_citation():
    import pytest

    with pytest.raises(Exception):
        Claim(statement="x", citations=[])


def test_investment_thesis_minimum():
    t = InvestmentThesis(
        company="ACME",
        summary="ok",
        recommendation="HOLD",
    )
    assert t.recommendation == "HOLD"
    assert t.strengths == []


def test_legal_risk_report():
    r = LegalRiskReport(
        matter="m",
        summary="s",
        overall_risk="LOW",
        risks=[
            LegalRisk(
                title="t",
                severity="LOW",
                description="d",
                citations=[Citation(source_id="abc", quote="q")],
            )
        ],
    )
    assert r.risks[0].severity == "LOW"
