# Technical Document Intelligence — RAG

> **Status: early-stage foundation.** Stages 1-24 of a 60-stage plan are
> complete. File discovery and loading (TXT, Markdown and PDF) and splitting
> that text into page- and section-attributed chunks now work and are tested;
> the embedding layer (contract, OpenAI adapter, batching and bounded
> retries, and the chunk→vector embed step) is built and tested offline but
> **not yet wired into a pipeline**, and retrieval and answer-generation are
> **designed but not yet implemented**. No benchmark results exist yet.
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

**Implemented and tested (Stages 1-24):**

- ✅ Requirements analysis and modular architecture (`ARCHITECTURE.md`)
- ✅ Git repository with a clean, ignored working tree
- ✅ Package scaffold for every planned component
- ✅ Windows virtualenv with 18 pinned direct dependencies
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
- ✅ A sample corpus generated on demand, byte-identical on every run, split
  into readable documents and files that deliberately fail — so one command
  demonstrates loaded, skipped, failed and unsupported results together
- ✅ Command-line entry points (`python -m scripts.cli prepare|ingest`) that
  print the report as text or JSON and tell invocation failures (status 1)
  apart from files that failed to load (status 3)
- ✅ Chunking: loaded documents split into overlapping chunks by LangChain's
  recursive character splitter, each stamped with a stable `doc_id:index` id
  and carrying the `page` (PDF) or `section` (Markdown heading) it came from,
  so a retrieved passage can later be cited to its source
- ✅ An embedder contract before any provider: a structural `Embedder`
  interface whose rules — one vector per chunk in order, one shared
  dimension, queries and documents in the same vector space, finite
  components, blank input rejected locally with a stable error code — are
  enforced by a contract suite that any future implementation must pass,
  alongside an offline test double that makes retrieval testable with no
  network calls
- ✅ The OpenAI embedding adapter: it builds its client from settings (the
  key is checked before any client exists), validates every response before
  a caller sees it — count, dimension consistency, finite components — and
  translates provider failures into the shared error hierarchy while never
  copying the provider's message, so a credential cannot reach logs; all of
  it proven through a stubbed delegate, with no network call required
- ✅ Batched embedding calls behind one bounded retry policy: texts are split
  into order-preserving batches, transient provider failures (429, 5xx,
  dropped connections) are retried with capped exponential backoff and
  jitter, while a wrong key or a bad request fails immediately — and each
  retry logs only the attempt number, wait and error *type*, never the
  provider's message
- ✅ The embed step that joins chunks to vectors: one vector per chunk in
  order, cheap seam guards (count and shared dimension) using the same
  error codes as the adapter, provider failures passing through untouched,
  and a counts-only log line — proven both with a deliberately
  contract-breaking embedder double and with the real adapter across
  batches
- ✅ Idempotent re-ingestion: the ingest pipeline decides before doing —
  an unchanged file never reaches the chunker, embedder or store, while
  an edited one replaces via delete-by-`doc_id` then upsert so no stale
  chunk (including the tail of a shortened document) can linger; every
  fallible step runs before that one destructive write, so a failed
  embed leaves the previous version intact instead of vanished, and a
  missing file comes back as a typed result rather than an exception
- ✅ Persistent vector indexing over Chroma: one cosine collection whose
  rows carry the chunk id, text and the metadata schema, upserts that
  replace rather than append so the overwrite protocol writes through to
  real storage, a delete that reports its removed count, and chroma's
  raw exceptions translated into the shared error hierarchy — proven by
  running the ingest pipeline twice against a real on-disk index and
  reopening it with a second instance to show the data persists
- ✅ **859 tests passing**, with the chunking, embedding, pipeline and
  storage modules and the CLI measured at 100% branch-inclusive coverage
  alongside the rest of the pipeline modules

**Planned — not yet implemented:**

- ⬜ ChromaDB query path, document registry and a CLI index command (Stages 25–29)
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

### Run the pipeline from the command line

The repository ships no documents — `data/` is git-ignored, and checked-in
binary fixtures are unreviewable. Generate them instead: the generator is a
fixed set of literals, so every run writes byte-identical files and re-ingest
never mistakes a regeneration for an edit.

```powershell
python -m scripts.cli prepare    # write the sample corpus into data/samples
python -m scripts.cli ingest     # read it back and print the ingest report
```

`data/samples/` holds two areas, and the split is deliberate:

| Directory | Contents |
|---|---|
| `corpus/` | four readable documents plus a 3-page PDF with extractable text — the material a later retrieval stage will be evaluated against |
| `edge-cases/` | a blank file, a scanned PDF with no text layer, a password-protected PDF, a non-UTF-8 export and an unsupported image |

One run therefore produces every outcome the ingest service reports: **5
loaded, 2 skipped, 2 failed, 1 unsupported**.

Exit statuses are what a caller acts on:

| Status | Meaning |
|---|---|
| `0` | ran, and no file failed |
| `1` | the command could not run — a missing directory, a bad setting |
| `2` | the arguments themselves were wrong (argparse's convention) |
| `3` | the report printed, but some files failed to load |

Status 1 and 3 are kept apart on purpose: one says your invocation is wrong,
the other says your data is wrong, and the report still prints on `3` so
there is something to act on.

Both commands accept `--format json`, which puts only JSON on stdout and
keeps logs and hints on stderr, so the output can be piped. `ingest` also
takes `--no-recursive`; `prepare` takes `--force` to overwrite samples you
have edited, and leaves them alone without it.

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
python -m mypy app scripts       # type check
python -m pytest                 # tests
python -m pytest --cov=app --cov=scripts --cov-report=term-missing   # tests + coverage
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
