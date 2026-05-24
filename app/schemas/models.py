"""Domain models: typed chunks, citations, structured analyst outputs."""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Source / chunk model
# ---------------------------------------------------------------------------


class ModalityType(StrEnum):
    PDF_TEXT = "pdf_text"
    PDF_FIGURE = "pdf_figure"
    IMAGE = "image"
    EXCEL_TABLE = "excel_table"


class SourceLocator(BaseModel):
    """Precise pointer back into the original artifact."""

    file_name: str
    page: int | None = None  # PDF page (1-based)
    bbox: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    sheet: str | None = None  # Excel sheet name
    cell_range: str | None = None  # e.g. "A1:D20"
    image_id: str | None = None  # for standalone images / extracted figures


class Chunk(BaseModel):
    """Indexable unit with full provenance."""

    id: str
    modality: ModalityType
    text: str  # for vision content this is the model-generated description
    locator: SourceLocator
    extra: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    source_id: str = Field(..., description="Chunk id this claim is grounded in.")
    quote: str = Field(..., description="Verbatim or near-verbatim supporting text.")

    @field_validator("quote")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Citation quote must not be empty.")
        return v


class Claim(BaseModel):
    statement: str
    citations: list[Citation] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    grounding_score: float | None = Field(default=None, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


# --- Investment thesis ---


class FinancialMetric(BaseModel):
    name: str
    value: str
    period: str | None = None
    citations: list[Citation] = Field(min_length=1)


class InvestmentThesis(BaseModel):
    company: str
    summary: str
    recommendation: Literal["BUY", "HOLD", "SELL", "INSUFFICIENT_EVIDENCE"]
    key_metrics: list[FinancialMetric] = Field(default_factory=list)
    strengths: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    catalysts: list[Claim] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# --- Legal risk report ---


class LegalRisk(BaseModel):
    title: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    affected_parties: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(min_length=1)
    mitigation: str | None = None


class LegalRiskReport(BaseModel):
    matter: str
    summary: str
    overall_risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"]
    risks: list[LegalRisk] = Field(default_factory=list)
    obligations: list[Claim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# --- Verification result ---


class GuardrailReport(BaseModel):
    schema_ok: bool
    citations_ok: bool
    grounding_ok: bool
    numeric_ok: bool
    issues: list[str] = Field(default_factory=list)
    grounding_scores: dict[str, float] = Field(default_factory=dict)
