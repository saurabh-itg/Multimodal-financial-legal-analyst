"""Pull the open-source Ollama models the analyst expects.

Idempotent: ``ollama pull`` is a no-op when a model is already present locally.

Usage::

    python scripts/setup_ollama.py
    python scripts/setup_ollama.py --host http://localhost:11434
    python scripts/setup_ollama.py --skip-vision     # skip the big vision model
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable

import httpx

from app.core.config import get_settings

DEFAULT_MODELS = ("qwen2.5-coder:7b", "minicpm-v", "nomic-embed-text")
VISION_MODELS = {"minicpm-v", "llava", "bakllava", "llama3.2-vision:11b"}


def _api_root(host_or_base: str) -> str:
    """Strip a trailing ``/v1`` so we can hit the native Ollama API."""
    return host_or_base.rstrip("/").removesuffix("/v1")


def _wait_for_server(root: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{root}/api/tags", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.0)
    raise SystemExit(
        f"Ollama server not reachable at {root!r}: {last_err}\n"
        "Start it with `ollama serve` (or `docker compose up ollama`)."
    )


def _existing_models(root: str) -> set[str]:
    r = httpx.get(f"{root}/api/tags", timeout=10.0)
    r.raise_for_status()
    return {m["name"] for m in r.json().get("models", [])}


def _pull(root: str, model: str) -> None:
    print(f"  -> pulling {model} ...", flush=True)
    with httpx.stream(
        "POST",
        f"{root}/api/pull",
        json={"name": model, "stream": True},
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        last_status = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                import json

                evt = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            status = evt.get("status", "")
            if status and status != last_status:
                print(f"     {status}", flush=True)
                last_status = status
            if evt.get("error"):
                raise RuntimeError(evt["error"])


def setup(host: str, models: Iterable[str]) -> None:
    root = _api_root(host)
    print(f"Using Ollama at {root}")
    _wait_for_server(root)
    have = _existing_models(root)
    for model in models:
        if model in have:
            print(f"  [ok] {model} already present")
            continue
        _pull(root, model)
    print("All models ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama server URL (default: OLLAMA_BASE_URL from .env)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Override the model list (repeatable). Defaults to all three.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip pulling the vision model (saves ~8GB).",
    )
    args = parser.parse_args(argv)

    host = args.host or get_settings().ollama_base_url
    models = args.models or list(DEFAULT_MODELS)
    if args.skip_vision:
        models = [m for m in models if m not in VISION_MODELS]
    setup(host, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
