# Multimodal Financial & Legal Analyst Agent

AI agent that ingests **PDF reports, chart/figure images, and Excel sheets**
simultaneously and produces a **structured investment thesis or legal risk report** with
**source attribution (citations)** and **hallucination guardrails**.

> Built for senior-level use cases in finance and legal: every claim is grounded in a
> retrievable source artifact, validated by an NLI-based grounding check, and emitted as
> a Pydantic-validated JSON contract.

---

## Highlights

| Capability | Implementation |
|---|---|
| Multimodal ingestion | PyMuPDF (PDF text + embedded images), Pandas/openpyxl (XLSX), Vision LLM (charts) |
| Orchestration | LangGraph-style sequential pipeline with explicit state |
| Retrieval | ChromaDB persistent store, hybrid (BM25 + dense) reranking |
| Citations | Every fact carries `source_id`, `page`, `bbox` / `sheet!cell` / `image_id` |
| Guardrails | JSON schema validation, citation existence check, NLI grounding score, numeric consistency check |
| Outputs | `InvestmentThesis` or `LegalRiskReport` Pydantic models |
| Serving | FastAPI + Streamlit UI |
| Ops | Docker, docker-compose, structured logging, OpenTelemetry hooks, pytest, ruff, GitHub Actions |

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │              Streamlit UI / cURL             │
                └──────────────────────┬───────────────────────┘
                                       │ multipart upload
                ┌──────────────────────▼───────────────────────┐
                │               FastAPI Service                │
                └──────────────────────┬───────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   ┌──────────┐                 ┌────────────┐                  ┌──────────┐
   │ PDF       │                 │ Excel       │                  │ Image    │
   │ Loader    │                 │ Loader      │                  │ Loader   │
   │ (PyMuPDF) │                 │ (pandas)    │                  │ (vision) │
   └────┬──────┘                 └─────┬───────┘                  └────┬─────┘
        │ TextChunk, FigureRef         │ TableChunk                    │ ImageChunk
        └────────────────┬─────────────┴─────────────┬─────────────────┘
                         ▼                           ▼
                 ┌──────────────────┐       ┌──────────────────┐
                 │ Embedder + Chroma │       │ Cite-able Asset  │
                 │ vector store      │       │ Registry         │
                 └─────────┬─────────┘       └────────┬─────────┘
                           └──────────┬───────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Orchestrator (Graph) │
                          │ plan → retrieve →    │
                          │ analyze → draft →    │
                          │ verify → finalize    │
                          └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ Guardrails:          │
                          │  - schema            │
                          │  - citation exists   │
                          │  - NLI grounding     │
                          │  - numeric check     │
                          └──────────┬───────────┘
                                     ▼
                       Structured Report (JSON + Markdown)
```

---

## Quickstart (fully local, open-source models via Ollama)

### 1. Install Python deps
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

### 2. Install Ollama and pull models
Install [Ollama](https://ollama.com/download), make sure `ollama serve` is running,
then pull the three open-source models the app uses:
```powershell
python scripts/setup_ollama.py
# or, equivalently:
ollama pull qwen2.5-coder:7b
ollama pull minicpm-v   # optional; skip with --skip-vision
ollama pull nomic-embed-text
```

### 3. Run API
```powershell
uvicorn app.api.main:app --reload --port 8000
```

### 4. Run UI
```powershell
streamlit run app/ui/streamlit_app.py
```

### 5. Or run everything (Ollama + API + UI) with Docker
```powershell
docker compose up --build
```
The `ollama-pull` init container automatically downloads the three default
models on first boot, then exits. The API waits for it to finish.

### 6. Try it
```powershell
curl -X POST http://localhost:8000/v1/analyze `
  -F "mode=investment" `
  -F "files=@samples/sample_10k.pdf" `
  -F "files=@samples/financials.xlsx" `
  -F "files=@samples/revenue_chart.png"
```

### Switching providers
The same code path works against any OpenAI-compatible endpoint. Set
`LLM_PROVIDER=openai` (or `azure`) in `.env`, fill in `OPENAI_API_KEY` /
`OPENAI_BASE_URL`, and pick model names appropriate for that provider
(`gpt-4o-mini`, `text-embedding-3-small`, ...).

---

## Project Layout

```
app/
  api/              FastAPI routes & schemas
  core/             config, logging, telemetry
  ingestion/        PDF, Excel, image loaders → typed chunks
  retrieval/        embeddings, vector store, hybrid retriever
  orchestrator/     graph nodes: plan, retrieve, analyze, draft, verify
  guardrails/       citation check, NLI grounding, numeric consistency
  schemas/          Pydantic output contracts (InvestmentThesis, LegalRiskReport)
  llm/              provider-agnostic LLM + vision client
  ui/               Streamlit frontend
tests/              unit + integration tests
samples/            example inputs
```

---

## Configuration

All settings via `.env` (see `.env.example`). Key vars:
- `LLM_PROVIDER` (`ollama` (default) | `openai` | `azure` | `anthropic`)
- `LLM_MODEL` (default `qwen2.5-coder:7b`)
- `LLM_VISION_MODEL` (default `minicpm-v`)
- `EMBEDDING_MODEL` (default `nomic-embed-text`)
- `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL` (only used when `LLM_PROVIDER != ollama`)
- `CHROMA_PERSIST_DIR`
- `GROUNDING_MIN_SCORE` (default 0.55)
- `MAX_FILE_MB`

### Recommended open-source models
| Role | Model | Approx. size |
|---|---|---|
| Chat / reasoning | `qwen2.5-coder:7b` | 4.7 GB |
| Vision | `minicpm-v` | ~1.6 GB |
| Embeddings | `nomic-embed-text` | 274 MB |

You can swap in any other Ollama-served model (e.g. `qwen2.5:14b`, `mistral`,
`llava:13b`, `mxbai-embed-large`) by setting the corresponding env var.

---

## Guardrails in Detail

1. **Schema validation** — output must parse into `InvestmentThesis` / `LegalRiskReport`.
2. **Citation existence** — every `Citation.source_id` must exist in the asset registry.
3. **NLI grounding** — each claim is scored against the cited evidence text via a
   cross-encoder (`cross-encoder/nli-deberta-v3-base`); claims below `GROUNDING_MIN_SCORE`
   are flagged or removed.
4. **Numeric consistency** — numbers in claims (e.g. "$4.2B revenue") are checked against
   a regex extraction over the cited evidence; mismatches are flagged.
5. **Refusal-on-empty** — if no grounded claims survive, the agent refuses rather than
   hallucinating.

---

## License
MIT
