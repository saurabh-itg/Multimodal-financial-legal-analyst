"""Shared test fixtures: stub the LLM, embedder, and vision client to avoid network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """Replace network-bound LLM calls with deterministic stubs."""

    from app import llm

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Tiny deterministic embedding: first 8 char codes / 256
        out = []
        for t in texts:
            v = [0.0] * 16
            for i, ch in enumerate(t[:16]):
                v[i] = (ord(ch) % 64) / 64.0
            out.append(v)
        return out

    def fake_describe_image(path, prompt):
        return f"[image description for {Path(path).name}]"

    def fake_chat_json(system: str, user: str, **_):
        # Minimal valid investment-thesis-shaped JSON for unit tests
        # Pull a chunk id out of evidence block if present.
        import re

        ids = re.findall(r"\[id=([^\]]+)\]", user)
        cid = ids[0] if ids else "missing"
        if "Sheet" in user or "EVIDENCE" in user:
            return {
                "company": "ACME Corp",
                "summary": "Stub summary.",
                "recommendation": "HOLD",
                "key_metrics": [],
                "strengths": [
                    {
                        "statement": "Revenue grew.",
                        "citations": [{"source_id": cid, "quote": "Revenue grew."}],
                        "confidence": 0.7,
                    }
                ],
                "risks": [],
                "catalysts": [],
                "overall_confidence": 0.6,
            }
        return {"queries": ["revenue", "risks", "growth", "margins"]}

    def fake_chat_text(system, user, **_):
        return "stub"

    monkeypatch.setattr(llm.client, "embed", fake_embed)
    monkeypatch.setattr(llm.client, "describe_image", fake_describe_image)
    monkeypatch.setattr(llm.client, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm.client, "chat_text", fake_chat_text)
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(llm, "describe_image", fake_describe_image)
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm, "chat_text", fake_chat_text)

    # Also patch in already-imported modules
    import app.ingestion.image_loader as img_mod
    import app.ingestion.pdf_loader as pdf_mod
    import app.orchestrator.pipeline as pipe_mod
    import app.retrieval.hybrid as hyb_mod

    monkeypatch.setattr(img_mod, "describe_image", fake_describe_image)
    monkeypatch.setattr(pdf_mod, "describe_image", fake_describe_image)
    monkeypatch.setattr(hyb_mod, "embed", fake_embed)
    monkeypatch.setattr(pipe_mod, "chat_json", fake_chat_json)


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield tmp_path / "chroma"
    get_settings.cache_clear()  # type: ignore[attr-defined]
