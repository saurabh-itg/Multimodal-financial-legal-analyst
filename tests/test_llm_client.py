"""Tests for the provider-agnostic LLM client.

These tests stub the underlying ``OpenAI`` SDK so nothing hits the network.
They confirm that:

* The client is configured for Ollama by default (open-source models).
* JSON extraction tolerates common open-source-model output quirks
  (markdown fences, leading prose, trailing commentary).
* Switching ``LLM_PROVIDER`` between ``ollama`` and ``openai`` flips the
  base URL and API key the SDK is constructed with.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings, get_settings
from app.llm import _extract_json
from app.llm import client as llm_client


@pytest.fixture(autouse=True)
def _stub_llm():
    """Override the conftest-level LLM stubs.

    The shared ``_stub_llm`` fixture in ``tests/conftest.py`` swaps
    ``chat_json`` / ``embed`` / ``describe_image`` for canned versions so the
    high-level pipeline tests can run offline. For *this* file we want the
    real implementations so we can exercise their fallbacks; we just need to
    stub the underlying OpenAI client itself.
    """
    yield


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    get_settings.cache_clear()  # type: ignore[attr-defined]
    llm_client.reset_client()
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]
    llm_client.reset_client()


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


def test_default_provider_is_ollama_with_open_source_models(monkeypatch):
    # Make sure no env vars override our defaults during this test.
    for var in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_VISION_MODEL",
        "EMBEDDING_MODEL",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.llm_model == "qwen2.5-coder:7b"
    assert s.llm_vision_model == "minicpm-v"
    assert s.embedding_model == "nomic-embed-text"
    assert s.ollama_base_url.endswith("/v1")
    assert s.effective_api_key == "ollama"
    assert s.effective_base_url == s.ollama_base_url


def test_openai_provider_routes_to_openai_creds(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    s = Settings(_env_file=None)
    assert s.effective_api_key == "sk-test"
    assert s.effective_base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"queries": ["a", "b"]}',
        '```json\n{"queries": ["a", "b"]}\n```',
        'Sure! Here is the JSON:\n```\n{"queries": ["a", "b"]}\n```',
        'Here you go: {"queries": ["a", "b"]}.  Hope that helps!',
    ],
)
def test_extract_json_recovers_common_oss_outputs(raw):
    parsed = _extract_json(raw)
    assert parsed == {"queries": ["a", "b"]}


def test_extract_json_handles_braces_inside_strings():
    raw = 'noise {"text": "value with } brace", "n": 1} trailing'
    assert _extract_json(raw) == {"text": "value with } brace", "n": 1}


def test_extract_json_raises_on_garbage():
    import json

    with pytest.raises(json.JSONDecodeError):
        _extract_json("nothing json shaped here")


# ---------------------------------------------------------------------------
# Client wiring (provider switch + chat_json fallback)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str, *, refuse_response_format: bool = False) -> None:
        self._content = content
        self._refuse_response_format = refuse_response_format
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._refuse_response_format and "response_format" in kwargs:
            raise RuntimeError("model does not support response_format")
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, content: str, *, refuse_response_format: bool = False) -> None:
        self.completions = _FakeChatCompletions(
            content, refuse_response_format=refuse_response_format
        )
        self.chat = _FakeChat(self.completions)


def test_chat_json_falls_back_when_response_format_unsupported(monkeypatch):
    fake = _FakeClient(
        '```json\n{"queries": ["x"]}\n```',
        refuse_response_format=True,
    )
    monkeypatch.setattr(llm_client, "_client", lambda: fake)

    out = llm_client.chat_json("sys", "user")
    assert out == {"queries": ["x"]}
    # First call attempted with response_format, second succeeded without it.
    assert len(fake.completions.calls) == 2
    assert "response_format" in fake.completions.calls[0]
    assert "response_format" not in fake.completions.calls[1]


def test_chat_json_uses_configured_chat_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    fake = _FakeClient('{"queries": ["q"]}')
    monkeypatch.setattr(llm_client, "_client", lambda: fake)

    llm_client.chat_json("sys", "user")
    assert fake.completions.calls[0]["model"] == "qwen2.5:14b"


def test_describe_image_degrades_gracefully_when_vision_model_missing(monkeypatch, tmp_path):
    from PIL import Image

    img = tmp_path / "x.png"
    Image.new("RGB", (50, 50), "white").save(img)

    class _BoomCompletions:
        def create(self, **_):
            raise RuntimeError("model not found")

    class _BoomClient:
        chat = type("Chat", (), {"completions": _BoomCompletions()})()

    monkeypatch.setattr(llm_client, "_client", lambda: _BoomClient())
    out = llm_client.describe_image(img, "describe")
    assert "vision model unavailable" in out
    assert "x.png" in out
