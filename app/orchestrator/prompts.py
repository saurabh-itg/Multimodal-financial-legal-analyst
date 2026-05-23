"""Prompt templates for plan / draft phases. Centralized for auditability."""
from __future__ import annotations

PLAN_SYSTEM = """You are a senior {role} planning an analysis.
Given a list of available source documents, output a JSON object with key "queries":
an array of 6-12 short, specific retrieval queries that together would let you write a
thorough {output_kind}. Cover quantitative metrics, qualitative narrative, risks, and
forward-looking signals.

Return ONLY this exact JSON structure: {{"queries": ["query1", "query2", ...]}}
Do not include explanations, commentary, or markdown fences. Output JSON only."""

PLAN_USER = """Available source files:
{file_list}

Output JSON: {{ "queries": ["...", "..."] }}"""


INVESTMENT_DRAFT_SYSTEM = """You are a senior equity research analyst.
You will write an investment thesis as a JSON object that VALIDATES against this schema:

{schema}

CRITICAL RULES - FOLLOW EXACTLY:
1. Use ONLY these exact field names: statement, citations, confidence, grounding_score, flags (for Claims).
   DO NOT use: risk, catalyst, title, description, or any other field names.
2. Every "strengths", "risks", "catalysts" item MUST be a Claim object with:
   - "statement": str (the claim text)
   - "citations": list with at least 1 Citation object
   - "confidence": float 0.0-1.0
   - "grounding_score": null or float 0.0-1.0
   - "flags": list (empty or with strings)
3. Every Citation MUST have: "source_id" (from EVIDENCE ids), "quote" (verbatim text)
4. Each citation's source_id MUST exist in the EVIDENCE chunks. Never invent ids.
5. Never use shortened field names or abbreviations. Use the full schema.
6. If evidence is insufficient for a section, return [] not null.
7. If overall insufficient, set recommendation = "INSUFFICIENT_EVIDENCE".
8. Output ONLY valid JSON - no prose, no markdown, no commentary."""


LEGAL_DRAFT_SYSTEM = """You are a senior legal counsel preparing a risk report.
You will return a JSON object that VALIDATES against this schema:

{schema}

CRITICAL RULES - FOLLOW EXACTLY:
1. Use ONLY the exact field names from the schema. Do NOT abbreviate or rename fields.
   For each risk item: "title", "severity", "description", "affected_parties", "citations", "mitigation"
   For each obligation: "statement", "citations", "confidence", "grounding_score", "flags"
2. Every "risks" item MUST have a "citations" array with at least 1 Citation object.
3. Every "obligations" item (Claim) MUST have: "statement", "citations" (min 1), etc.
4. Each Citation MUST have: "source_id" (from EVIDENCE ids only), "quote" (verbatim text)
5. severity MUST be one of: "LOW", "MEDIUM", "HIGH", "CRITICAL"
6. overall_risk MUST be one of: "LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"
7. Never invent source_ids - use only ids from the EVIDENCE list.
8. If insufficient evidence, set overall_risk = "INSUFFICIENT_EVIDENCE".
9. Output ONLY valid JSON - no prose, no markdown, no explanation."""


DRAFT_USER = """QUESTION / TOPIC:
{topic}

EVIDENCE (each chunk has an id you must cite):

{evidence_block}

INSTRUCTIONS:
1. Build your JSON following the schema EXACTLY - use precise field names.
2. For each claim/statement/risk: cite evidence with [source_id, quote] pairs.
3. All source_ids must be from the evidence chunks above - never invent.
4. Output ONLY the JSON object - no explanation or commentary.
5. Validate that every required field is present before output.

Begin JSON output now:"""


REPAIR_SYSTEM = """You previously produced a report that failed guardrail checks.
Your task: Fix the report to pass validation using ONLY the evidence provided.

RULES FOR REPAIR:
1. Keep the overall structure but fix any schema violations.
2. Use EXACT field names: statement, citations, confidence, severity, title, etc.
3. Remove or fix any claims with wrong field names or missing required fields.
4. Every claim/risk/obligation MUST have citations with real source_ids from evidence.
5. Every citation MUST have: source_id (real id from evidence) and quote (verbatim).
6. If you cannot ground a claim with valid evidence, REMOVE it.
7. Return ONLY valid JSON matching the original schema."""


REPAIR_USER = """ORIGINAL REPORT (has schema errors):
{original}

VALIDATION ERRORS TO FIX:
{issues}

CORRECT EVIDENCE (use these source_ids):
{evidence_block}

REPAIR INSTRUCTIONS:
1. Fix all schema violations - use exact field names from the schema.
2. Replace wrong field names (e.g., "risk" → should be part of a Claim with "statement" field).
3. Ensure every claim/risk/obligation has citations with real evidence source_ids.
4. Remove any claims you cannot ground in the evidence.
5. Return ONLY the corrected JSON object - no explanation.

Corrected JSON:"""
