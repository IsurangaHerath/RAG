# Technical Document Intelligence — RAG

> **Status: early-stage foundation.** Stages 1–16 of a 60-stage plan are
> complete. File discovery and loading (TXT, Markdown and PDF) now work and
> are tested; the rest of the ingestion, retrieval and answer-generation
> pipeline is **designed but not yet implemented**, and no benchmark results
> exist yet.
> See [Project status](#project-status) for exactly what does and does not work.

A Retrieval-Augmented Generation (RAG) system for technical documents: upload
PDFs, TXT and Markdown files, then ask questions in natural language and get
answers that are **grounded in those documents** — with citations back to the
source file and page, and an explicit "I don't have enough information" when
the corpus does not contain the answer.

---

## Why this project

Language models answer from general training knowledge. That is fine for
general knowledge and wrong for *your* documents: internal specifications,
manuals, reports and standards that the model has never seen. Retrieval-Augmented
Generation fixes this by fetching the relevant passages first and requiring the
model to answer from them.

The hard part — and what this project is really about — is making the answers
**verifiable**. Every claim should trace to a stored chunk, and the system
should refuse to answer rather than invent. That is why the design puts
citation validation and abstention handling at the centre rather than treating
them as extras.

## Architecture overview

```
  Upload                          Query
    │                               │
    ▼                               ▼
┌─────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│  FastAPI │──▶│  Ingestion   │──▶│  Chunking & │──▶│   ChromaDB   │
│  routes  │   │ load·dedup   │   │  Embeddings │   │ vector store │
└─────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                          │ top-k
  answer ◀──┌──────────────┐◀──┌──────────────┐◀──────────┘
  + sources │  Generation  │   │  Retrieval   │
            │ prompts·LLM  │   │ filter·score │
            └──────────────┘   └──────────────┘
```

**Design principle:** each arrow is a typed interface, not a direct import.
Loaders, embedders, the vector store, the reranker and the LLM are each behind
a small `Protocol`, so any one of them can be replaced — a different embedding
provider, a different vector database, a local model instead of OpenAI —
without touching the rest of the pipeline.

Full detail lives in **[ARCHITECTURE.md](ARCHITECTURE.md)**: data flows,
component responsibilities and interfaces, security notes, and risk/cost analysis.

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | modern typing (`Self`, `Never`), broad library support |
| Web API | FastAPI + Uvicorn | async, Pydantic-native validation, auto OpenAPI docs |
| Orchestration | LangChain 1.x | splitters, chat models, vector-store integrations |
| LLM | OpenAI API | `gpt-4o-mini` for answers, `text-embedding-3-large` for embeddings |
| Vector store | ChromaDB | persistent local HNSW index, zero external services |
| Config | pydantic-settings | typed `.env`, secrets never logged |
| Evaluation | pandas, scikit-learn | precision/recall/MRR computed and reproducible |
| Tests | pytest, pytest-cov | offline by default — no API key needed |
| Quality | Ruff, mypy (strict) | linting, formatting, static type checking |
| Packaging | Docker | reproducible runtime |

## Project status

**Implemented and tested (Stages 1–16):**

- ✅ Requirements analysis and modular architecture (`ARCHITECTURE.md`)
- ✅ Git repository with a clean, ignored working tree
- ✅ Package scaffold for every planned component
- ✅ Windows virtualenv with 17 pinned direct dependencies
- ✅ Centralized, validated configuration (`.env.example` → `.env`)
- ✅ Secret handling: `SecretStr` + tests proving the key never reaches a log
- ✅ One shared error hierarchy mapped to HTTP status codes
- ✅ Lint / format / type-check / test workflow in one script, plus CI
- ✅ Document domain model: stable `doc_id`, content hashes, chunk IDs
- ✅ Recursive file discovery that blocks path traversal and symlink escapes
- ✅ TXT and Markdown loaders: strict UTF-8, BOM handling, line-ending
  normalisation, formatting preserved byte-for-byte
- ✅ PDF loading with page numbers kept for citations, per-page failure
  isolation, and specific errors for encrypted or corrupt files
- ✅ Document metadata extraction: sanitised PDF info (title, author, dates,
  keywords), titles derived from headings or filenames, and flat mappings
  safe for ChromaDB
- ✅ Ingest service that runs discovery → load → metadata across a whole
  directory: empty files and scans reported as *skipped*, unreadable ones as
  *failed* with the typed error's code, one bad file never stopping the batch
- ✅ Deduplication: comparing a file's path-id and content-hash against what
  is already known classifies it as new, unchanged, modified or renamed, so
  edits replace their chunks and identical content is never indexed twice
- ✅ Typed `DocumentLoadError` for unreadable, undecodable and mislabelled files
- ✅ Per-batch ingest report grouping problems by their error code with the
  files that produced them, plus bytes read and optional dedup tallies —
  renderable as plain text or JSON, and never carrying document text
- ✅ Structured logging that cannot leak document text: a key deny-list, a
  value-size guard, and type-name collapsing for objects whose `repr` quotes
  their own content, off only when `LOG_DOCUMENT_CONTENT` is deliberately set
- ✅ **649 tests passing**, core modules at 100% branch coverage

**Planned — not yet implemented:**

- ⬜ Sample data preparation and CLI entry points (Stage 17)
- ⬜ Chunking and embedding generation (Stages 18–23)
- ⬜ ChromaDB persistent indexing and a CLI index command (Stages 24–29)
- ⬜ Semantic retrieval, filtering and thresholds (Stages 30–36)
- ⬜ Grounded answer generation with citations (Stages 37–44)
- ⬜ Evaluation metrics and configuration experiments (Stages 45–52)
- ⬜ Query and document-management API endpoints (Stages 53–57)
- ⬜ Docker image and Compose deployment (Stages 58–60)

**Not yet measured:** retrieval accuracy, answer quality, latency and API cost.
No figures are quoted anywhere in this repository because none have been
measured. Numbers will be added only after they come from a reproducible run
of the evaluation suite.

## Getting started

### Prerequisites

- Python 3.12 (`python --version`)
- Git
- An OpenAI API key — **only** needed for live LLM calls, which the test
  suite does not make

### Installation (Windows / PowerShell)

```powershell
git clone https://github.com/IsurangaHerath/RAG.git
cd RAG

# create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install runtime + development dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

# confirm everything is set up
.\scripts\check.ps1
```

If PowerShell blocks activation, run once and reopen the terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

`cmd.exe` users: use `.venv\Scripts\activate.bat` instead of `Activate.ps1`.

### Configuration

```powershell
Copy-Item .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`.env` is git-ignored; `.env.example` holds placeholders only. Settings are
read into a single typed `Settings` object — no module reads `os.environ`
directly. Validation runs at startup, so a bad value fails immediately with a
message naming the field instead of failing mid-request.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(required for live calls)* | OpenAI credential, stored as `SecretStr` |
| `CHAT_MODEL` | `gpt-4o-mini` | model that writes answers |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | model that indexes documents |
| `DATA_DIR` | `./data` | uploads, samples and ChromaDB state |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | splitting behaviour |
| `TOP_K` | `5` | chunks retrieved per question |
| `MAX_UPLOAD_MB` / `MAX_QUERY_CHARS` | `20` / `4000` | request limits |
| `LOG_LEVEL` | `INFO` | logging verbosity |
| `LOG_DOCUMENT_CONTENT` | `False` | keep off — `True` writes document text to logs |

Full list in [`.env.example`](.env.example).

## Development workflow

One command runs every check, stopping at the first failure:

```powershell
.\scripts\check.ps1              # format check → lint → mypy → pytest + coverage
.\scripts\check.ps1 -Fix         # auto-format and auto-fix lint issues first
.\scripts\check.ps1 -SkipTypes   # faster inner loop
```

Or run each tool yourself:

```powershell
python -m ruff format .          # apply formatting
python -m ruff check .           # lint
python -m mypy app               # type check
python -m pytest                 # tests
python -m pytest --cov=app --cov-report=term-missing   # tests + coverage
```

### Conventions

- **Types:** `mypy --strict` on `app/`. Public functions are annotated.
- **Errors:** raise a subclass of `app.domain.errors.RAGError`, which carries a
  stable `code` and an `http_status`. Put developer context in `detail`, which
  is excluded from responses by default.
- **Tests:** unit tests must run offline. Anything that calls the real OpenAI
  API is marked `@pytest.mark.live` and stays deselected.
- **Secrets:** never hardcode a key, never commit `.env`.
- **Docs:** update `ARCHITECTURE.md` in the same change as the behaviour it describes.

### Testing philosophy

The suite runs with **no API key and no network**: OpenAI is mocked, ChromaDB
runs against a temporary directory. That keeps tests free, fast and honest —
a green run never depends on a paid call succeeding. Live integration tests
exist as a separate, opt-in layer.

## Roadmap

Development follows 60 numbered stages grouped into 10 phases:

| Phase | Stages | Scope |
|---|---|---|
| 1 | 1–8 | planning, repository, environment, configuration, quality, README |
| 2 | 9–17 | document discovery, loaders, metadata, ingestion, dedup |
| 3 | 18–23 | chunking and embedding generation |
| 4 | 24–29 | persistent ChromaDB indexing |
| 5 | 30–36 | retrieval engine and diagnostics |
| 6 | 37–44 | grounded generation with citations |
| 7 | 45–52 | evaluation metrics and experiments |
| 8 | 53–57 | REST API |
| 9 | 58–59 | Docker and reproducible deployment |
| 10 | 60 | final validation and release |

Each stage is pushed as its own commit. See [`ARCHITECTURE.md` §13](ARCHITECTURE.md)
for the implementation order.

## License

MIT — see [`pyproject.toml`](pyproject.toml).

Sample documents used for evaluation will be either self-authored or
redistributable; no copyrighted datasets will be committed.
