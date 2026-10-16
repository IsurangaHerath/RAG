# Technical Document Intelligence RAG — Architecture Design

Status: design only (no application code)
Stack: Python 3.12 · FastAPI · LangChain · OpenAI API · ChromaDB · pytest · Docker

---

## 1. Objectives

- Ingest technical documents (PDF, DOCX, TXT, MD, HTML) and make them searchable by meaning, not just keywords.
- Answer natural-language questions **grounded** in retrieved document content, with per-claim citations back to `document : page/chunk`.
- Expose the capability as a versioned REST API suitable for a web UI or other services.
- Keep every stage swappable (loader, chunker, embedder, vector store, reranker, LLM) behind small interfaces so providers can change without touching business logic.
- Ship with tests, containerized deployment, and measurable answer quality.

## 2. Scope / Features

**In scope**
- Multipart document upload, dedup by content hash, async ingestion with status reporting.
- Parsing → chunking → embedding → indexing pipeline with retry/backoff and per-document failure isolation.
- Semantic retrieval with metadata filters, top-k + MMR diversity, optional reranking.
- Grounded generation: context-only answers, mandatory citations, explicit "insufficient context" behavior, optional SSE streaming.
- Observability: structured logs, request IDs, token/latency metrics; health/readiness endpoints.
- Evaluation harness: golden Q/A set scored for faithfulness and retrieval recall.

**Out of scope (v1)**
- OCR of scanned images, multi-hop agentic reasoning, cross-lingual translation, real-time collaborative editing, fine-tuning.

## 3. High-Level Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │                 Docker network               │
  ┌────────┐  HTTPS    │  ┌─────────────┐        ┌─────────────────┐   │
  │ Client │──────────▶│  │  api        │        │ worker          │   │
  │ (UI/   │  JSON/SSE │  │  FastAPI    │        │ ingest tasks    │   │
  │  CLI)  │◀──────────│  │  routers +  │───────▶│ (Background/    │   │
  └────────┘           │  │  Pydantic   │ enqueue│  arq/Celery)    │   │
                       │  └──────┬──────┘        └───────┬─────────┘   │
                       │         │ read                   │ write      │
                       │         ▼                        ▼            │
                       │  ┌─────────────┐        ┌─────────────────┐   │
                       │  │ registry    │        │ ChromaDB        │   │
                       │  │ (SQLite/PG) │        │ persistent vol  │   │
                       │  └─────────────┘        └────────┬────────┘   │
                       │         ▲                        │            │
                       │         │ upload metadata        │            │
                       │  ┌──────┴──────┐                 │            │
                       │  │ object store│  data/uploads   │            │
                       │  └─────────────┘                 │            │
                       └───────────┬──────────────────────┬────────────┘
                                   │ OpenAI API           │
                                   │ (embeddings + chat)  │
                                   ▼                      │
                              ┌─────────┐                 │
                              │ OpenAI  │◀────────────────┘
                              └─────────┘
```

Two planes:

- **Control plane (api)**: accepts uploads, serves query answers, exposes status. Never blocks on long ingestion.
- **Data plane (worker)**: performs parse → chunk → embed → upsert; owns writes to Chroma.
- **Query path** runs in the api process (embed query → search → generate), because it is latency-bound, not throughput-bound.

## 4. Package Layout

```
RAG/
├── app/
│   ├── __init__.py
│   ├── config.py            # Stage 06: pydantic-settings, typed config, secrets from env
│   ├── main.py              # Stage 53: app factory, middleware, router registration
│   ├── domain/
│   │   ├── models.py        # Stage 09: Document, Chunk, RetrievedChunk, Citation, Answer
│   │   └── errors.py        # Stage 07: error taxonomy (UploadError, ParseError, ...)
│   ├── ingestion/           # Stages 10–16
│   │   ├── discovery.py     # find supported files, block path traversal
│   │   ├── loaders.py       # TXT/MD/PDF → raw text + page metadata
│   │   ├── metadata.py      # normalise filename/type/source path
│   │   ├── deduplication.py # stable IDs + content hashes
│   │   ├── service.py       # orchestrates discovery → load → metadata
│   │   └── reporting.py     # structured per-batch ingest report
│   ├── preprocessing/
│   │   └── chunking.py      # Stage 18: LangChain recursive text splitting
│   ├── embeddings/          # Stages 20–22
│   │   ├── base.py          # provider-agnostic Embedder interface
│   │   ├── openai_embedder.py  # OpenAI adapter + validation
│   │   └── batching.py      # batched calls, bounded retries/backoff
│   ├── storage/             # Stages 24–27
│   │   ├── chroma.py        # VectorIndex impl over persistent Chroma
│   │   └── registry.py      # DocumentRegistry (status, checksum)
│   ├── retrieval/           # Stages 30–35
│   │   ├── retriever.py     # top-k semantic search
│   │   ├── scoring.py       # distance → score conversion (justified)
│   │   ├── filters.py       # validated metadata filtering
│   │   ├── thresholds.py    # optional score threshold
│   │   └── diagnostics.py   # dev-only retrieval inspector (not routed)
│   ├── generation/          # Stages 37–43
│   │   ├── llm_client.py    # ChatOpenAI wrapper, error mapping
│   │   ├── prompts.py       # grounding rules, untrusted-input delimiters
│   │   ├── chain.py         # validate → retrieve → assemble → generate
│   │   └── citations.py     # parse/validate citations, abstention handling
│   ├── evaluation/          # Stages 45–52
│   │   ├── metrics.py       # P@k, R@k, F1@k, RR/MRR, nDCG
│   │   └── runner.py        # experiment runner + CSV/Markdown reports
│   ├── api/
│   │   ├── schemas.py       # Stage 43: Pydantic request/response contracts
│   │   ├── deps.py          # Stage 53: DI providers ( Depends(...) )
│   │   └── routes/          # Stages 53–55
│   │       ├── health.py    # /healthz (liveness) and /readyz (dependencies)
│   │       ├── query.py     # POST /v1/query
│   │       └── documents.py # list / index / delete documents
│   └── observability/
│       ├── logging.py       # Stage 16: structured logs, no document content
│       └── metrics.py       # Stage 51: latency and token usage
├── tests/
│   ├── unit/                # offline tests, mocked OpenAI (no paid calls)
│   └── integration/         # real Chroma on tmp_path, mocked HTTP
├── scripts/                 # Stage 17: sample-data preparation, CLI entry points
├── docs/                    # sample reports, evaluation notes
├── data/                    # git-ignored local data (uploads, Chroma persist dir)
├── docker/                  # Stage 58: container helpers
├── docker-compose.yml       # Stage 59
├── Dockerfile               # Stage 58
├── pyproject.toml           # Stage 07: ruff + mypy + pytest config
├── requirements.txt         # Stage 05: pinned dependencies
├── README.md                # Stage 08
├── ARCHITECTURE.md          # this file
└── .env.example             # Stage 06: placeholders only, never real secrets
```

Stage 04 created only the packages, `__init__.py` files, and a structural
guard test. Implementation modules are added by their own stages, so no
empty placeholder files exist yet.


Dependency rule: `api → (retrieval, generation, storage) → domain`; `ingestion → storage → domain`; `domain` imports nothing from the app. Infrastructure modules implement `domain`-level Protocols.

## 5. Component Responsibilities & Interfaces

Interfaces are Python `typing.Protocol` contracts (signatures only — implementation deferred).

### 5.1 Domain models (`domain/models.py`)

```python
@dataclass(frozen=True)
class Document:      doc_id: str; filename: str; media_type: str
                     sha256: str; created_at: datetime; metadata: dict
@dataclass(frozen=True)
class Chunk:         chunk_id: str; doc_id: str; text: str
                     page: int | None; section: str | None; metadata: dict
@dataclass(frozen=True)
class RetrievedChunk: chunk: Chunk; score: float; source: str      # "chroma" | "rerank"
@dataclass(frozen=True)
class Citation:      doc_id: str; chunk_id: str; page: int | None; quote: str
@dataclass(frozen=True)
class Answer:        text: str; citations: list[Citation]
                     usage: TokenUsage; model: str; latency_ms: int
                     grounded: bool                                # passed grounding check
class IngestStatus:  PENDING | PARSING | INDEXING | INDEXED | FAILED
```

### 5.2 Ingestion

Responsibility: turn a stored file into indexed, citable chunks. Idempotent per `doc_id` (content-hash IDs; re-upload overwrites rather than duplicates).

```python
class DocumentLoader(Protocol):
    def load(self, path: Path, *, doc_id: str, media_type: str) -> list[LCDocument]: ...
        # LCDocument = langchain_core.documents.Document (page_content + metadata incl. page_no)

class Chunker(Protocol):
    def split(self, documents: list[LCDocument], *, doc_id: str) -> list[Chunk]: ...
        # RecursiveCharacterSplitter(1000, 150); markdown/table-aware for structured text;
        # emits chunk_id = f"{doc_id}:{index}", preserves page/section metadata

class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
        # batched (e.g. 128), retries with exponential backoff on 429/5xx
    def embed_query(self, text: str) -> list[float]: ...

class IngestPipeline(Protocol):
    def run(self, doc_id: str) -> IngestResult: ...
        # load → chunk → embed → upsert → registry.update_status(INDEXED | FAILED)
```

### 5.3 Storage

Responsibility: persistence + retrieval of vectors; lifecycle state of documents. Two independent stores behind interfaces so they can be swapped (Chroma → pgvector; SQLite → Postgres).

```python
class VectorIndex(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[vector]) -> None: ...
    def similarity_search(self, query: str, *, k: int,
                          filter: MetadataFilter | None = None) -> list[RetrievedChunk]: ...
        # MetadataFilter e.g. {"doc_id": {"$in": [...]}, "section": "Requirements"}
    def delete_document(self, doc_id: str) -> int: ...          # returns chunks removed
    def count(self) -> int: ...

class DocumentRegistry(Protocol):
    def create(self, doc: Document) -> None: ...
    def get(self, doc_id: str) -> DocumentRecord | None: ...
    def list(self, *, status: IngestStatus | None, limit: int, offset: int) -> list[DocumentRecord]: ...
    def update_status(self, doc_id: str, status: IngestStatus, error: str | None = None) -> None: ...
    def find_by_hash(self, sha256: str) -> DocumentRecord | None: ...   # dedup

class FileStore(Protocol):
    def save(self, stream: BinaryIO, *, doc_id: str) -> Path: ...
    def path_for(self, doc_id: str) -> Path: ...
```

Chroma specifics: single persistent collection `techdocs` (HNSW, cosine), `persist_directory` on a Docker volume in single-node mode or Chroma server mode if the api/worker split needs a network share. Metadata schema: `doc_id, chunk_id, filename, page, section, media_type, ingested_at`.

### 5.4 Retrieval

Responsibility: fetch the *smallest sufficient* evidence set within the model's context budget.

```python
class QueryAnalyzer(Protocol):
    def analyze(self, query: str, history: list[Turn]) -> RetrievalQuery: ...
        # optional: rewrite follow-ups ("what about v2?") into standalone query; extract filters

class Retriever(Protocol):
    def retrieve(self, rq: RetrievalQuery, *, k: int = 20) -> list[RetrievedChunk]: ...
        # embed_query → chroma top-k (with filter) → MMR dedupe

class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]: ...
        # no-op implementation in v1; cross-encoder or LLM-based later (cost/latency trade-off)

class ContextBuilder(Protocol):
    def build(self, query: str, chunks: list[RetrievedChunk], *, budget_tokens: int
             ) -> GroundedContext: ...
        # truncates to budget, orders by doc/page, emits blocks:
        # [1] filename.pdf p.12 (doc_abc:17) "…verbatim text…"   and remembers the mapping
```

### 5.5 Generation

Responsibility: produce an answer that can be *verified* against the context.

```python
class PromptBuilder(Protocol):
    def build(self, query: str, ctx: GroundedContext, history: list[Turn]) -> list[BaseMessage]: ...

class AnswerGenerator(Protocol):
    def generate(self, query: str, ctx: GroundedContext, history: list[Turn],
                 *, stream: bool = False) -> Answer | AsyncIterator[str]: ...
        # LangChain ChatOpenAI (e.g. gpt-4o-mini default, gpt-4o configurable),
        # temperature 0, response_format json for citation extraction
        # retries: none on 4xx, exponential backoff on 429/5xx, request timeout budget

class GroundingChecker(Protocol):
    def check(self, answer: str, ctx: GroundedContext) -> bool: ...
        # every cited chunk_id exists in ctx; optional claim-level entailment via cheap LLM call
```

**Grounding contract enforced by the system prompt**: answer only from `<context>`; cite `[n]` markers for every factual sentence; if the context does not contain the answer, reply with the fixed insufficient-information phrase (surfaced as `grounded=False`). Citations are then validated programmatically — a citation to an unknown chunk is an error, not a hallucination that leaks through.

### 5.6 API layer

Responsibility: transport, validation, auth, rate limits, and dependency wiring only — no business logic.

| Route | Method | Request | Response |
|---|---|---|---|
| `/v1/documents` | POST | multipart `file`, optional `metadata` JSON | `202 {doc_id, status:"pending"}` |
| `/v1/documents/{id}` | GET | — | `200 DocumentStatus` (status, pages, chunks, error) |
| `/v1/documents/{id}` | DELETE | — | `204` (removes file + registry row + Chroma chunks) |
| `/v1/documents` | GET | `status, limit, offset` | `200 Page[DocumentStatus]` |
| `/v1/query` | POST | `{question, doc_ids?, top_k?, stream?}` | `200 Answer` or `text/event-stream` |
| `/healthz` / `/readyz` | GET | — | liveness; readiness checks Chroma + OpenAI reachability |

Cross-cutting middleware: request ID, structured access log, auth (API key / JWT), body-size limit, per-key rate limit, exception → error taxonomy → RFC7807-style JSON (`{code, message, request_id}`).

### 5.7 Workers / queue

```python
class TaskQueue(Protocol):
    def enqueue(self, task: IngestTask) -> str: ...
    def add_handler(self, name: str, fn: Callable[[IngestTask], None]) -> None: ...
```
v1: FastAPI `BackgroundTasks` or `arq`+Redis (chosen so tests can run an inline fake queue). Celery is a drop-in later if throughput demands it. Task carries `doc_id` only; worker re-reads the file from `FileStore`.

### 5.8 Configuration (single source of truth)

`pydantic-settings` reads env: `OPENAI_API_KEY`, `EMBEDDING_MODEL` (default `text-embedding-3-large`), `CHAT_MODEL` (default `gpt-4o-mini`), `CHROMA_PATH`/`CHROMA_HOST`, `CHUNK_SIZE`, `TOP_K`, `RERANK_TOP_N`, `CONTEXT_BUDGET_TOKENS`, `MAX_UPLOAD_MB`, `AUTH_MODE`, `WORKER_MODE` (`inline|queue`).

## 6. Data Flow

### 6.1 Upload → index

```
Client            api                 FileStore   Registry   Queue    worker            Chroma
  │ POST /v1/documents (multipart)
  │──────────────▶│ validate (type, size, magic bytes)
  │               │ sha256 → dedup hit? → 409/return existing doc_id
  │               │ save file ──────────▶ data/uploads/{doc_id}/…
  │               │ create row ─────────▶ status=PENDING
  │               │ enqueue(ingest) ────────────────────────────▶│
  │◀── 202 {doc_id} ──│                                         │
  │               │                                        loader.load()
  │               │                                        chunker.split()
  │               │                                        embedder.embed_documents()
  │               │                                              │ upsert(ids, vecs, metas)
  │               │                                              │───────────▶ collection
  │               │                                        registry.status=INDEXED
  │ GET /v1/documents/{id} ──▶ 200 {status:"indexed", chunks: 412, pages: 38}
```

Failure at any stage → `status=FAILED` with `error` (e.g. `PARSE_ERROR: encrypted pdf`), other documents unaffected. Re-processing the same `doc_id` overwrites chunks (delete-by-`doc_id` then upsert) — ingestion is idempotent.

### 6.2 Query → grounded answer

```
Client                api (query path)                         OpenAI
  │ POST /v1/query {question, doc_ids?}
  │──────────────────▶│ auth, validate, request_id
  │                   │ QueryAnalyzer.analyze(question, history)   # rewrite follow-up
  │                   │ Embedder.embed_query(standalone_query) ─────────────▶ embeddings
  │                   │ VectorIndex.similarity_search(k=20, filter={doc_id…}) ◀── Chroma
  │                   │ MMR dedupe → Reranker.rerank(top_n=6)        [optional]
  │                   │ ContextBuilder.build(budget=3–6k tokens)
  │                   │ PromptBuilder.build(system: grounding rules + context + citations)
  │                   │ AnswerGenerator.generate ────────────────────────▶ chat.completions
  │                   │◀────────────────────────────────────────────── answer + usage
  │                   │ citations.parse → GroundingChecker.check
  │                   │ persist usage metrics (model, tokens, latency)
  │◀── 200 Answer {text, citations[], grounded, usage, sources[]} ──────│
```

Sources are returned *unconditionally*, even when the model declines to answer, so the UI can show "closest passages" and the user can judge.

## 7. Dependencies

### 7.1 Installed now (Stage 05 — verified importable)

| Layer | Package | Pinned version | Purpose |
|---|---|---|---|
| RAG core | `langchain` | 1.4.6 | agent/chain orchestration |
| | `langchain-core` | 1.6.11 | prompts, documents, Runnables |
| | `langchain-openai` | 1.7.2 | `ChatOpenAI`, `OpenAIEmbeddings` |
| | `langchain-text-splitters` | 1.1.3 | `RecursiveCharacterTextSplitter` |
| | `langchain-chroma` | 1.1.0 | Chroma as a LangChain vector store |
| LLM | `openai` | 3.31.0 | provider SDK |
| Vector DB | `chromadb` | 1.5.9 | persistent HNSW index |
| Web | `fastapi` | 0.143.3 | REST API |
| | `uvicorn[standard]` | 0.54.0 | ASGI server |
| | `pydantic` | 2.14.1 | request/response validation |
| Data science | `numpy` | 2.5.4 | vector maths |
| | `pandas` | 3.0.6 | experiment tables |
| | `scikit-learn` | 1.9.1 | evaluation metrics |
| Test | `pytest` | 9.1.1 | test runner |

Environment: **Python 3.12.2**, Windows, virtualenv at `.venv/`.

`requirements.txt` pins direct dependencies with `==`.
`requirements.lock.txt` records all 122 transitive packages for exact
reproduction.

### 7.2 Installation commands (Windows / PowerShell)

```powershell
cd D:\Projects\RAG

# 1. create the virtual environment (once)
python -m venv .venv

# 2. activate it
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks activation, run once, then re-open:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. verify
python -m pytest tests/ -q
```

For an exact reproduction of the tested environment:

```powershell
python -m pip install -r requirements.lock.txt
```

cmd.exe users: activate with `.venv\Scripts\activate.bat` instead.

**Always run tools from inside the venv.** Prefixing with
`.\.venv\Scripts\python.exe -m pytest` avoids accidentally using a global
interpreter that lacks the dependencies.

### 7.3 Planned additions (not yet installed)

| Layer | Package | Needed from stage | Purpose |
|---|---|---|---|
| Web | `python-multipart` | 55 | multipart document upload |
| Config | `pydantic-settings` | 06 | validated `.env` settings |
| Docs | `pypdf`, `python-docx`, `beautifulsoup4` | 11–12 | PDF/DOCX/HTML parsing |
| Ops | `structlog`, `tenacity` | 16 / 21 | logging, bounded retries |
| Test | `pytest-cov`, `pytest-asyncio` | 07 | coverage, async tests |
| Quality (opt.) | `ragas`, `tiktoken`, `sentence-transformers` | 45–52 / 31 | eval, token budgeting, rerank |

LangChain 1.x moved classes that lived in `langchain` during 0.x, so all
LangChain imports stay isolated to adapter modules (`ingestion/*`,
`retrieval/*`, `generation/*`) and are asserted by
`tests/unit/test_environment.py`. Upgrades should touch one layer only.


## 8. Testing Strategy (pytest)

| Layer | What | How |
|---|---|---|
| Unit | chunker boundaries/overlap, context budgeting, citation parsing, grounding check, error mapping | pure functions, `FakeEmbedder`, no network |
| Contract | each `Protocol` implementation conforms; Pydantic schemas don't break | `pytest` parametrized over implementations |
| Integration | upload→ingest→query happy path + failure path | real Chroma on tmp_path, `MockTransport` for OpenAI (recorded fixtures), inline queue |
| E2E | docker-compose stack, one real OpenAI call | marked `@pytest.mark.e2e`, opt-in in CI (secret-gated) |
| Evaluation | faithfulness, answer relevancy, retrieval recall@k | golden `tests/evaluation/dataset.jsonl` + RAGAS; run nightly, fail on regression > threshold |

Coverage gate ≥ 80% on `app/domain`, `app/retrieval/context.py`, `app/generation/citations.py`. CI: lint (ruff) → typecheck (mypy) → unit+integration → build image.

## 9. Deployment (Docker)

- **Multi-stage Dockerfile**: builder installs from lockfile with `--no-cache`; runtime stage is `python:3.12-slim`, non-root user, `uvicorn app.main:app --workers N`.
- **`docker-compose.yml` services**: `api` (ports 8000, env, healthcheck `/healthz`), `worker` (same image, different command), `chroma` (official image, volume `chroma_data`), `redis` (optional), `upload_data` volume. `.env.example` documents every variable; `OPENAI_API_KEY` injected, never baked into the image.
- Scaling: stateless `api` replicas behind a reverse proxy; single writer to Chroma in v1 (worker) to avoid index contention.

## 10. Security & Privacy

- AuthN/AuthZ: API keys or JWT on all `/v1/*` routes; per-key rate limit and upload quota.
- Upload hardening: allow-list of extensions **and** magic-byte sniffing, max size (e.g. 50 MB), filename sanitization, store outside the web root, zip-bomb/decompression limits, virus scan hook (ClamAV) for untrusted corpora.
- Secrets: env/secret manager only; never logged. Redact `Authorization`, prompt bodies, and document text from logs.
- Data residency: document content is sent to OpenAI for embeddings and generation — disclose in the data-flow docs; if unacceptable, the `Embedder`/`AnswerGenerator` Protocols allow a local-model implementation with no code changes elsewhere. Optional zero-retention endpoints for enterprise accounts.
- Prompt injection: document text is untrusted input — delimit it, instruct the model to ignore instructions inside context, and never let tool/registry actions be driven by model output in v1.
- Isolation: non-root container, read-only FS except data volumes, no shell in production image, dependency scanning (pip-audit/trivy) in CI.

## 11. Risks & Cost Levers

| Risk | Mitigation |
|---|---|
| Hallucinated citations | programmatic grounding check; reject/flag non-conforming answers; golden-set eval in CI |
| Chunking hurts recall (tables, specs) | layout-aware PDF parsing, page metadata, hybrid (BM25+vector) and reranking as config flags; eval tracks recall@k |
| LangChain API churn | isolate imports in adapter modules; pin versions |
| OpenAI outage/rate limit | tenacity backoff, timeouts, circuit breaker, configurable alternate model |
| Cost blow-up | token budget caps, cache common embeddings (Chroma already stores them; add query-result caching), default `gpt-4o-mini`, rerank off by default |
| Large-file OOM | stream parsing, per-file size/page limits, batched embedding |
| Chroma contention (multi-writer) | single-writer worker in v1; server mode + sharding later |

**Indicative cost** (orders of magnitude, depends on corpus): embedding `text-embedding-3-large` ≈ $0.13/M tokens → a 1,000-doc (≈300-page) corpus ≈ $0.20–1 to index. Answers with `gpt-4o-mini` (≈$0.15/M in, $0.60/M out) ≈ $0.0003–0.001 per 5k-context query; 10k queries/day ≈ $3–10/day. Reranking with a frontier model or cross-encoder GPU adds latency and, if LLM-based, cost — off by default. Use `gpt-4o` only for hard reasoning questions (config flag).

## 12. Acceptance Criteria (design-level DoD)

1. `docker compose up` starts api, worker, chroma; `/readyz` green.
2. Uploading a 100-page technical PDF returns `202` and reaches `status=indexed` without blocking the request; re-uploading the same file does not duplicate chunks.
3. `POST /v1/query` answers from the corpus with ≥2 citations whose `doc_id`/`chunk_id` all resolve to stored chunks; a question outside the corpus returns the fixed insufficient-information response.
4. Deleting a document removes its file, registry row, and all its chunks (verified by a query that no longer returns them).
5. All non-network code paths covered by unit/integration tests in CI; no real API key needed except the opt-in e2e/eval jobs.
6. No secrets, raw document text, or prompts appear in logs; container runs as non-root.
7. Answer quality measured on the golden set (faithfulness ≥ threshold, recall@5 ≥ threshold) with a CI regression gate.

## 13. Implementation Plan (no code yet — execution order)

1. **Scaffold & contracts** — repo layout, `pyproject.toml`, ruff/mypy/pytest config, `domain/models.py` + Protocols, `.env.example`, CI pipeline skeleton.
2. **Storage vertical slice** — `FileStore`, `DocumentRegistry`, Chroma `VectorIndex` adapter + contract tests.
3. **Ingestion path** — loaders → chunker → embedder → pipeline, inline worker execution, upload/download/status endpoints; verify end-to-end with a sample PDF (mocked OpenAI in tests).
4. **Retrieval path** — query embedding, top-k + filter + MMR, `ContextBuilder` with token budget; golden "retrieval-only" tests.
5. **Generation path** — prompts with grounding rules, `AnswerGenerator`, citation parsing + grounding check, `/v1/query` (JSON then SSE streaming).
6. **Async worker** — TaskQueue abstraction + arq/Redis implementation, failure handling, status transitions, idempotent re-ingest, `DELETE` cascade.
7. **Hardening** — auth, rate limit, upload validation, error taxonomy, structured logging, metrics, health checks.
8. **Evaluation & ops** — golden dataset, RAGAS scoring, Dockerfile/compose hardening, load test (e.g. 50 concurrent queries), README runbook.

Each step ends green in CI and demo-able; steps 2–5 form the walking skeleton before any hardening work begins.
