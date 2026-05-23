"""Provider-agnostic LLM + vision + embeddings client.

Defaults to a local **Ollama** server running open-source models, but works with
any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, vLLM, LM Studio, ...).

Why one client for everything?
    Ollama exposes an OpenAI-compatible REST API at ``/v1`` so the same
    ``openai.OpenAI`` SDK can drive both. We just swap ``base_url`` and the
    default model names.

Why custom JSON extraction?
    Open-source models honour ``response_format={"type": "json_object"}`` less
    reliably than GPT-4-class models. They sometimes wrap output in
    ```` ```json ... ``` ```` fences or emit a leading sentence. The
    :func:`_extract_json` helper recovers from those cases so the pipeline
    stays robust across model sizes.
"""
from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Create (and cache) a single OpenAI-compatible client.

    The client is cached because instantiating it spins up an httpx connection
    pool; we want one pool per process, not one per request.
    """
    s = get_settings()
    kwargs: dict[str, Any] = {"api_key": s.effective_api_key}
    base_url = s.effective_base_url
    if base_url:
        kwargs["base_url"] = base_url
    if s.llm_provider == "ollama":
        # Local CPU inference can be slow; keep the connection alive for a long
        # time so the first cold-start request doesn't time out.
        kwargs["timeout"] = httpx.Timeout(s.ollama_request_timeout, connect=10.0)
    log.info(
        "llm.client.init",
        provider=s.llm_provider,
        base_url=base_url or "<openai-default>",
        chat_model=s.llm_model,
        vision_model=s.llm_vision_model,
        embedding_model=s.embedding_model,
    )
    return OpenAI(**kwargs)


def reset_client() -> None:
    """Drop the cached client. Used by tests that mutate env vars.

    Defensive: in tests ``_client`` may have been monkeypatched to a plain
    callable that doesn't carry an ``lru_cache`` wrapper.
    """
    cache_clear = getattr(_client, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON parse for raw LLM output.

    Tries, in order:
        1. ``json.loads`` on the raw string
        2. The first ```` ```json ... ``` ```` fenced block
        3. The first balanced ``{...}`` substring

    Raises :class:`json.JSONDecodeError` if all strategies fail.
    """
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty content", text, 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(text)
    if fence:
        candidate = fence.group("body").strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: scan for the first balanced top-level object.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    return json.loads(candidate)

    raise json.JSONDecodeError("no JSON object found", text, 0)


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Chat completion that returns a parsed JSON object.

    Asks the server for ``response_format=json_object`` (Ollama, OpenAI and most
    OpenAI-compatible servers honour this) and falls back to a tolerant JSON
    extractor if the model emits prose or markdown fences anyway.
    """
    s = get_settings()
    client = _client()

    # Reinforce the JSON requirement in-band; some smaller open-source models
    # honour the prompt more reliably than the response_format flag.
    system_msg = system.rstrip() + (
        "\n\nReturn ONLY a single JSON object. "
        "No prose, no markdown code fences, no commentary."
    )

    create_kwargs: dict[str, Any] = {
        "model": model or s.llm_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user},
        ],
    }

    try:
        resp = client.chat.completions.create(
            **create_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as e:  # noqa: BLE001
        # Older / smaller Ollama models reject `response_format`. Retry once
        # without it; the system prompt still demands JSON.
        log.warning("llm.json_format_unsupported_retry_plain", error=str(e))
        resp = client.chat.completions.create(**create_kwargs)

    content = resp.choices[0].message.content or "{}"
    try:
        return _extract_json(content)
    except json.JSONDecodeError as e:
        log.error("llm.json_parse_failed", error=str(e), content=content[:500])
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def chat_text(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> str:
    s = get_settings()
    client = _client()
    resp = client.chat.completions.create(
        model=model or s.llm_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def describe_image(image_path: str | Path, prompt: str) -> str:
    """Use a vision model to extract a structured description of an image.

    Works with OpenAI's GPT-4o family **and** with Ollama vision models such as
    ``minicpm-v`` / ``llava`` / ``bakllava`` via the OpenAI-compatible API.
    """
    s = get_settings()
    client = _client()
    image_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    suffix = Path(image_path).suffix.lstrip(".").lower() or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    data_url = f"data:image/{suffix};base64,{b64}"

    try:
        resp = client.chat.completions.create(
            model=s.llm_vision_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous analyst. Describe the image factually. "
                        "If a chart, list axes, units, series, and visible numeric values. "
                        "Do not speculate beyond what is visible."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        # Vision models are large (~8GB+) and may not be installed locally.
        # Degrade gracefully so the pipeline can still run on text + tables.
        log.warning(
            "llm.vision_unavailable",
            error=str(e),
            model=s.llm_vision_model,
            file=str(image_path),
        )
        return f"[vision model unavailable; image '{Path(image_path).name}' not described]"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via the configured embedding model.

    For Ollama the recommended default is ``nomic-embed-text`` (768-dim);
    for OpenAI it's ``text-embedding-3-small``. Both expose the same
    ``/embeddings`` endpoint shape.
    """
    s = get_settings()
    client = _client()
    if not texts:
        return []
    cleaned = [t if t.strip() else " " for t in texts]
    resp = client.embeddings.create(model=s.embedding_model, input=cleaned)
    return [d.embedding for d in resp.data]
