"""End-to-end analysis pipeline.

Stages:
  1. ingest      - parse multimodal files into Chunks
  2. index       - hybrid retriever (Chroma + BM25)
  3. plan        - LLM proposes retrieval queries
  4. retrieve    - gather evidence chunks
  5. draft       - LLM emits structured JSON report
  6. verify      - guardrails (citations, NLI grounding, numerics)
  7. repair      - one-shot LLM repair if issues; then re-verify
  8. finalize    - annotate + return
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.core.logging import get_logger
from app.guardrails import annotate_claims, filter_low_grounding, verify_report
from app.ingestion import ingest_files
from app.llm import chat_json
from app.orchestrator.prompts import (
    DRAFT_USER,
    INVESTMENT_DRAFT_SYSTEM,
    LEGAL_DRAFT_SYSTEM,
    PLAN_SYSTEM,
    PLAN_USER,
    REPAIR_SYSTEM,
    REPAIR_USER,
)
from app.retrieval import HybridRetriever, Retrieved
from app.schemas import (
    Chunk,
    GuardrailReport,
    InvestmentThesis,
    LegalRiskReport,
)

log = get_logger(__name__)

Mode = Literal["investment", "legal"]


@dataclass
class AnalysisResult:
    mode: Mode
    report: InvestmentThesis | LegalRiskReport
    guardrails: GuardrailReport
    used_chunks: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    repaired: bool = False


def _build_evidence_block(retrieved: list[Retrieved]) -> str:
    blocks = []
    for r in retrieved:
        loc = r.chunk.locator
        loc_bits = [f"file={loc.file_name}"]
        if loc.page is not None:
            loc_bits.append(f"page={loc.page}")
        if loc.sheet:
            loc_bits.append(f"sheet={loc.sheet}")
        if loc.cell_range:
            loc_bits.append(f"cells={loc.cell_range}")
        if loc.image_id:
            loc_bits.append(f"image={loc.image_id}")
        header = f"[id={r.chunk.id}] [{r.chunk.modality.value}] " + " ".join(loc_bits)
        body = r.chunk.text.strip()
        if len(body) > 1500:
            body = body[:1500] + " …"
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _plan_queries(
    files: list[Path],
    *,
    mode: Mode,
    topic: str,
) -> list[str]:
    role = "equity research analyst" if mode == "investment" else "legal counsel"
    output_kind = "investment thesis" if mode == "investment" else "legal risk report"
    file_list = "\n".join(f"- {p.name}" for p in files)
    raw = chat_json(
        system=PLAN_SYSTEM.format(role=role, output_kind=output_kind),
        user=PLAN_USER.format(file_list=file_list) + f"\n\nTopic / question: {topic}",
    )
    qs = raw.get("queries", [])
    if not isinstance(qs, list) or not qs:
        qs = [topic]
    return [q for q in qs if isinstance(q, str) and q.strip()][:12]


def _retrieve(retriever: HybridRetriever, queries: list[str], *, per_query: int = 6) -> list[Retrieved]:
    seen: dict[str, Retrieved] = {}
    for q in queries:
        for r in retriever.retrieve(q, top_k=per_query):
            prev = seen.get(r.chunk.id)
            if prev is None or r.score > prev.score:
                seen[r.chunk.id] = r
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:30]


def _model_for(mode: Mode) -> type[InvestmentThesis | LegalRiskReport]:
    return InvestmentThesis if mode == "investment" else LegalRiskReport


def _normalize_claims(obj: dict | list | Any) -> dict | list | Any:
    """Recursively fix common field name mistakes in LLM output.
    
    This helps with models that use shortened/abbreviated field names instead of
    the correct schema field names (e.g., 'risk' → 'statement', 'title' alone → Claim object).
    """
    if isinstance(obj, dict):
        # Common renaming: model uses 'risk', 'catalyst', 'strength' instead of 'statement'
        if "risk" in obj and "statement" not in obj:
            obj["statement"] = obj.pop("risk")
        if "catalyst" in obj and "statement" not in obj:
            obj["statement"] = obj.pop("catalyst")
        if "strength" in obj and "statement" not in obj:
            obj["statement"] = obj.pop("strength")
        
        # If we have statement but no citations, try to extract from description/content
        if "statement" in obj and (not obj.get("citations") or obj.get("citations") == []):
            # If there's a description field, it might have citation info
            if "description" in obj and isinstance(obj.get("description"), str):
                obj["statement"] = obj["description"]
            # Ensure citations is a list
            if "citations" not in obj:
                obj["citations"] = []
        
        # Normalize nested structures
        for key, value in obj.items():
            if isinstance(value, (list, dict)):
                obj[key] = _normalize_claims(value)
    
    elif isinstance(obj, list):
        # Normalize each item
        return [_normalize_claims(item) for item in obj]
    
    return obj


def _draft(
    *,
    mode: Mode,
    topic: str,
    retrieved: list[Retrieved],
) -> InvestmentThesis | LegalRiskReport:
    Model = _model_for(mode)
    schema = json.dumps(Model.model_json_schema(), indent=2)
    system = (INVESTMENT_DRAFT_SYSTEM if mode == "investment" else LEGAL_DRAFT_SYSTEM).format(
        schema=schema
    )
    user = DRAFT_USER.format(topic=topic, evidence_block=_build_evidence_block(retrieved))
    raw = chat_json(system=system, user=user)
    # Normalize common field name mistakes before validation
    raw = _normalize_claims(raw)
    return Model.model_validate(raw)


def _repair(
    *,
    mode: Mode,
    original: InvestmentThesis | LegalRiskReport,
    issues: list[str],
    retrieved: list[Retrieved],
) -> InvestmentThesis | LegalRiskReport:
    Model = _model_for(mode)
    raw = chat_json(
        system=REPAIR_SYSTEM,
        user=REPAIR_USER.format(
            original=original.model_dump_json(indent=2),
            issues="\n".join(f"- {i}" for i in issues),
            evidence_block=_build_evidence_block(retrieved),
        ),
    )
    # Normalize common field name mistakes before validation
    raw = _normalize_claims(raw)
    return Model.model_validate(raw)


# ---------------------------------------------------------------------------


def run_analysis(
    files: list[str | Path],
    *,
    mode: Mode = "investment",
    topic: str = "",
) -> AnalysisResult:
    paths = [Path(p) for p in files]
    if not paths:
        raise ValueError("No files supplied.")

    log.info("pipeline.start", mode=mode, files=[p.name for p in paths])

    # 1-2. Ingest + index
    chunks: list[Chunk] = ingest_files(paths)
    if not chunks:
        raise RuntimeError("No content could be extracted from the provided files.")

    retriever = HybridRetriever()
    try:
        retriever.index(chunks)

        # 3. Plan
        topic = topic or (
            "Build a comprehensive investment thesis."
            if mode == "investment"
            else "Identify and explain all legal risks in these documents."
        )
        queries = _plan_queries(paths, mode=mode, topic=topic)
        log.info("pipeline.plan", queries=queries)

        # 4. Retrieve
        retrieved = _retrieve(retriever, queries)
        log.info("pipeline.retrieved", n=len(retrieved))

        # 5. Draft
        try:
            report = _draft(mode=mode, topic=topic, retrieved=retrieved)
        except ValidationError as ve:
            log.warning("pipeline.draft_invalid_schema", error=str(ve))
            # Treat as schema failure -> try one repair
            report = _repair(
                mode=mode,
                original=_model_for(mode).model_construct(),
                issues=[f"Schema error: {ve}"],
                retrieved=retrieved,
            )

        # 6. Verify
        guard = verify_report(report, retriever)
        annotate_claims(report, guard)

        repaired = False
        # 7. Repair if needed
        if guard.issues:
            log.info("pipeline.repairing", issues=len(guard.issues))
            try:
                repaired_report = _repair(
                    mode=mode, original=report, issues=guard.issues, retrieved=retrieved
                )
                guard2 = verify_report(repaired_report, retriever)
                annotate_claims(repaired_report, guard2)
                # accept repair only if it strictly reduced issues
                if len(guard2.issues) < len(guard.issues):
                    report, guard = repaired_report, guard2
                    repaired = True
            except (ValidationError, Exception) as e:  # noqa: BLE001
                log.warning("pipeline.repair_failed", error=str(e))

        # 8. Drop ungrounded claims as last line of defense
        removed = filter_low_grounding(report)
        if removed:
            log.info("pipeline.filtered_low_grounding", removed=removed)

        return AnalysisResult(
            mode=mode,
            report=report,
            guardrails=guard,
            used_chunks=[r.chunk.id for r in retrieved],
            queries=queries,
            repaired=repaired,
        )
    finally:
        retriever.cleanup()
