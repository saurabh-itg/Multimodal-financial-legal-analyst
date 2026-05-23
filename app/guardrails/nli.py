"""NLI-based grounding scorer using a HuggingFace cross-encoder.

Lazily loaded; if the model isn't available (offline / no torch), falls back to
a lexical-overlap proxy so the pipeline still returns meaningful scores.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _model():
    s = get_settings()
    if not s.enable_nli:
        return None
    try:
        from sentence_transformers import CrossEncoder

        log.info("nli.loading", model=s.nli_model)
        return CrossEncoder(s.nli_model)
    except Exception as e:  # noqa: BLE001
        log.warning("nli.unavailable_fallback_lexical", error=str(e))
        return None


_TOKEN_RE = re.compile(r"[A-Za-z0-9$%\.]+")


def _lexical_overlap(claim: str, evidence: str) -> float:
    a = {t.lower() for t in _TOKEN_RE.findall(claim)}
    b = {t.lower() for t in _TOKEN_RE.findall(evidence)}
    if not a:
        return 0.0
    return len(a & b) / len(a)


def grounding_score(claim: str, evidence: str) -> float:
    """Return entailment probability in [0,1]. Higher = better grounded."""
    if not claim.strip() or not evidence.strip():
        return 0.0
    model = _model()
    if model is None:
        return _lexical_overlap(claim, evidence)
    try:
        # Cross-encoder NLI: produces (entailment, neutral, contradiction) logits or single score
        scores = model.predict([(evidence, claim)])
        s = scores[0]
        try:
            # 3-class logits
            import numpy as np  # noqa: WPS433

            arr = np.asarray(s)
            if arr.ndim == 1 and arr.size == 3:
                exp = np.exp(arr - arr.max())
                probs = exp / exp.sum()
                # label 0 == entailment in deberta-nli
                return float(probs[0])
        except Exception:  # noqa: BLE001
            pass
        return float(s)
    except Exception as e:  # noqa: BLE001
        log.warning("nli.predict_failed_fallback", error=str(e))
        return _lexical_overlap(claim, evidence)
