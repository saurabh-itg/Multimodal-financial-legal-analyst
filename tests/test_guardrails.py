from app.guardrails import verify_report
from app.retrieval import HybridRetriever
from app.schemas import (
    Chunk,
    Citation,
    Claim,
    InvestmentThesis,
    ModalityType,
    SourceLocator,
)


def _retriever_with(chunks):
    r = HybridRetriever()
    r.index(chunks)
    return r


def test_citation_existence(tmp_chroma):
    chunk = Chunk(
        id="x1",
        modality=ModalityType.PDF_TEXT,
        text="Revenue grew 20% to $1.2B in 2023.",
        locator=SourceLocator(file_name="f.pdf", page=1),
    )
    r = _retriever_with([chunk])

    good = InvestmentThesis(
        company="ACME",
        summary="s",
        recommendation="HOLD",
        strengths=[
            Claim(
                statement="Revenue grew 20% to $1.2B in 2023.",
                citations=[Citation(source_id="x1", quote="Revenue grew 20% to $1.2B in 2023.")],
            )
        ],
    )
    g = verify_report(good, r)
    assert g.citations_ok is True

    bad = InvestmentThesis(
        company="ACME",
        summary="s",
        recommendation="HOLD",
        strengths=[
            Claim(
                statement="Revenue tripled.",
                citations=[Citation(source_id="does-not-exist", quote="x")],
            )
        ],
    )
    g2 = verify_report(bad, r)
    assert g2.citations_ok is False
    assert any("unknown source_id" in i for i in g2.issues)
    r.cleanup()


def test_numeric_consistency(tmp_chroma):
    chunk = Chunk(
        id="x2",
        modality=ModalityType.PDF_TEXT,
        text="Revenue was $1.2B.",
        locator=SourceLocator(file_name="f.pdf", page=1),
    )
    r = _retriever_with([chunk])
    rep = InvestmentThesis(
        company="ACME",
        summary="s",
        recommendation="HOLD",
        strengths=[
            Claim(
                statement="Revenue was $9.9B.",  # not in evidence
                citations=[Citation(source_id="x2", quote="Revenue was $1.2B.")],
            )
        ],
    )
    g = verify_report(rep, r)
    assert g.numeric_ok is False
    r.cleanup()
