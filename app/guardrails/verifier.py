"""Verifies a structured analyst report against its retrievable evidence."""
from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.guardrails.nli import grounding_score
from app.retrieval import HybridRetriever
from app.schemas import (
    Citation,
    Claim,
    GuardrailReport,
    InvestmentThesis,
    LegalRiskReport,
)

log = get_logger(__name__)

_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|bn|mn|m|b|k)?", re.IGNORECASE)


def _claims_of(report: InvestmentThesis | LegalRiskReport) -> list[tuple[str, list[Citation], str]]:
    """Returns list of (statement, citations, location_label)."""
    out: list[tuple[str, list[Citation], str]] = []
    if isinstance(report, InvestmentThesis):
        for m in report.key_metrics:
            out.append((f"{m.name}: {m.value}" + (f" ({m.period})" if m.period else ""),
                        m.citations, f"key_metrics.{m.name}"))
        for label in ("strengths", "risks", "catalysts"):
            for i, c in enumerate(getattr(report, label)):
                out.append((c.statement, c.citations, f"{label}[{i}]"))
    else:
        for i, r in enumerate(report.risks):
            out.append((f"{r.title}: {r.description}", r.citations, f"risks[{i}]"))
        for i, c in enumerate(report.obligations):
            out.append((c.statement, c.citations, f"obligations[{i}]"))
    return out


def _extract_numbers(text: str) -> set[str]:
    return {m.group(0).replace(" ", "").lower() for m in _NUM_RE.finditer(text)}


def verify_report(
    report: InvestmentThesis | LegalRiskReport,
    retriever: HybridRetriever,
) -> GuardrailReport:
    settings = get_settings()
    issues: list[str] = []
    grounding_scores: dict[str, float] = {}

    citations_ok = True
    grounding_ok = True
    numeric_ok = True

    valid_ids = set(retriever.all_ids())

    for statement, citations, label in _claims_of(report):
        # 1. Citation existence
        for cit in citations:
            if cit.source_id not in valid_ids:
                citations_ok = False
                issues.append(f"{label}: citation references unknown source_id={cit.source_id!r}")

        # 2. Grounding
        evidences = []
        for cit in citations:
            chunk = retriever.get(cit.source_id)
            if chunk is not None:
                evidences.append(chunk.text)
        evidence_text = "\n\n".join(evidences) if evidences else ""
        score = grounding_score(statement, evidence_text) if evidence_text else 0.0
        grounding_scores[label] = round(score, 3)
        if score < settings.grounding_min_score:
            grounding_ok = False
            issues.append(f"{label}: low grounding score {score:.2f} < {settings.grounding_min_score}")

        # 3. Numeric consistency
        claim_nums = _extract_numbers(statement)
        if claim_nums and evidence_text:
            ev_nums = _extract_numbers(evidence_text)
            missing = claim_nums - ev_nums
            if missing:
                numeric_ok = False
                issues.append(
                    f"{label}: numeric values not found in evidence: {sorted(missing)}"
                )

    rep = GuardrailReport(
        schema_ok=True,
        citations_ok=citations_ok,
        grounding_ok=grounding_ok,
        numeric_ok=numeric_ok,
        issues=issues,
        grounding_scores=grounding_scores,
    )
    log.info(
        "guardrails.verified",
        issues=len(issues),
        citations_ok=citations_ok,
        grounding_ok=grounding_ok,
        numeric_ok=numeric_ok,
    )
    return rep


def annotate_claims(
    report: InvestmentThesis | LegalRiskReport,
    guard: GuardrailReport,
) -> None:
    """Mutates the report to attach per-claim grounding scores and flags."""
    if isinstance(report, InvestmentThesis):
        groups: list[tuple[str, list[Claim]]] = [
            ("strengths", report.strengths),
            ("risks", report.risks),
            ("catalysts", report.catalysts),
        ]
    else:
        groups = [("obligations", report.obligations)]

    for label, items in groups:
        for i, c in enumerate(items):
            key = f"{label}[{i}]"
            if key in guard.grounding_scores:
                c.grounding_score = guard.grounding_scores[key]
                if c.grounding_score < get_settings().grounding_min_score:
                    c.flags.append("low_grounding")


def filter_low_grounding(
    report: InvestmentThesis | LegalRiskReport,
    min_score: float | None = None,
) -> int:
    """Drops claims below threshold. Returns number removed."""
    threshold = min_score if min_score is not None else get_settings().grounding_min_score
    removed = 0

    def _filter(items: list[Any]) -> list[Any]:
        nonlocal removed
        kept = []
        for c in items:
            score = getattr(c, "grounding_score", None)
            if score is None or score >= threshold:
                kept.append(c)
            else:
                removed += 1
        return kept

    if isinstance(report, InvestmentThesis):
        report.strengths = _filter(report.strengths)
        report.risks = _filter(report.risks)
        report.catalysts = _filter(report.catalysts)
    else:
        report.obligations = _filter(report.obligations)
    return removed
