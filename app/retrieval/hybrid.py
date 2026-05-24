"""Vector + BM25 hybrid retrieval over a per-request Chroma collection.

Collections are namespaced per analysis run so concurrent requests don't collide.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import embed
from app.schemas import Chunk

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Per-run hybrid retriever. Holds vector store + in-memory BM25."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        s = get_settings()
        self._client = chromadb.PersistentClient(
            path=s.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = f"{s.collection_name}_{self.run_id}"
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        self._chunk_index: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    def index(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [
            {
                "modality": c.modality.value,
                "file_name": c.locator.file_name,
                "page": c.locator.page or -1,
                "sheet": c.locator.sheet or "",
                "cell_range": c.locator.cell_range or "",
                "image_id": c.locator.image_id or "",
            }
            for c in chunks
        ]

        # Embed in batches
        batch = 64
        for i in range(0, len(texts), batch):
            sl = slice(i, i + batch)
            embs = embed(texts[sl])
            self._collection.add(
                ids=ids[sl],
                documents=texts[sl],
                embeddings=embs,
                metadatas=metadatas[sl],
            )

        for c in chunks:
            self._chunk_index[c.id] = c
            self._bm25_ids.append(c.id)
            self._bm25_corpus.append(_tokenize(c.text))

        self._bm25 = BM25Okapi(self._bm25_corpus)
        log.info("retriever.indexed", run_id=self.run_id, count=len(chunks))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, top_k: int = 8, alpha: float = 0.6) -> list[Retrieved]:
        """Hybrid retrieval. alpha weights dense; (1-alpha) weights BM25."""
        if not self._chunk_index:
            return []

        # Dense
        q_emb = embed([query])[0]
        dense = self._collection.query(
            query_embeddings=[q_emb], n_results=min(top_k * 3, len(self._chunk_index))
        )
        dense_ids = dense.get("ids", [[]])[0]
        dense_dists = dense.get("distances", [[]])[0]
        dense_scores: dict[str, float] = {}
        if dense_dists:
            mx = max(dense_dists) or 1.0
            for cid, d in zip(dense_ids, dense_dists, strict=False):
                dense_scores[cid] = 1.0 - (d / mx)  # higher = better

        # Sparse
        sparse_scores: dict[str, float] = {}
        if self._bm25 is not None:
            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens)
            mx = max(scores) if len(scores) else 0.0
            if mx > 0:
                for cid, sc in zip(self._bm25_ids, scores, strict=False):
                    sparse_scores[cid] = float(sc) / mx

        all_ids = set(dense_scores) | set(sparse_scores)
        merged = [
            (cid, alpha * dense_scores.get(cid, 0.0) + (1 - alpha) * sparse_scores.get(cid, 0.0))
            for cid in all_ids
        ]
        merged.sort(key=lambda x: x[1], reverse=True)
        out: list[Retrieved] = []
        for cid, score in merged[:top_k]:
            chunk = self._chunk_index.get(cid)
            if chunk is not None:
                out.append(Retrieved(chunk=chunk, score=score))
        return out

    # ------------------------------------------------------------------
    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunk_index.get(chunk_id)

    def all_ids(self) -> list[str]:
        return list(self._chunk_index.keys())

    def cleanup(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as e:  # noqa: BLE001
            log.warning("retriever.cleanup_failed", error=str(e))
