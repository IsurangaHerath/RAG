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
│   │   └── errors.py        # Stage 07: error taxonomy (UploadError, ParseError, DocumentLoadError, ...)
│   ├── ingestion/           # Stages 10–16, 23
│   │   ├── discovery.py     # find supported files, block path traversal
│   │   ├── loaders.py       # Stage 11: TXT/MD -> TextDocument; re-exports load_pdf
│   │   ├── pdf_loader.py    # Stage 12: PDF -> per-page text via pypdf
│   │   ├── metadata.py      # Stage 13: sanitised PDF info + derived metadata
│   │   ├── service.py       # Stage 14: orchestrates discovery → load → metadata
│   │   ├── deduplication.py # Stage 15: stable IDs + content hashes
│   │   ├── reporting.py     # Stage 16: structured per-batch ingest report
│   │   └── pipeline.py      # Stage 23: idempotent decide → chunk → embed → replace
│   ├── preprocessing/
│   │   └── chunking.py      # Stage 18: LangChain recursive text splitting
│   ├── embeddings/          # Stages 19–22
│   │   ├── base.py          # Stage 19: provider-agnostic Embedder interface
│   │   ├── openai_embedder.py  # Stage 20: OpenAI adapter + validation
│   │   ├── batching.py      # Stage 21: batched calls, bounded retries/backoff
│   │   └── generation.py    # Stage 22: chunks → aligned vectors (embed step)
│   ├── storage/             # Stages 24–27
│   │   ├── chroma.py        # Stage 24: VectorIndex write path over persistent Chroma
│   │   ├── registry.py      # Stage 25: DocumentRegistry (status, checksum) over SQLite
│   │   └── file_store.py    # Stage 26: FileStore (save/path_for) for stored source files
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
├── scripts/                 # helper tooling, kept as a package so tests can reuse it
│   ├── cli.py               # Stage 17: prepare / ingest entry points
│   ├── samples.py           # Stage 17: deterministic sample corpus
│   ├── pdf_fixtures.py      # Stage 17: PDF builders shared with the tests
│   └── check.ps1            # Stage 07: one-command quality gate
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

### 5.1 Domain models (`domain/models.py`) — implemented in Stage 09

Framework-free (no FastAPI/LangChain/pydantic imports). Immutable records.

```python
class IngestStatus(StrEnum):
    PENDING | PARSING | INDEXING | INDEXED | FAILED   # .is_terminal for the last two

@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str            # 'doc_' + 16 hex, derived from SOURCE PATH (see below)
    filename: str          # basename only, never a path
    media_type: str        # validated MIME type
    content: str           # extracted text; may be empty -> .is_empty
    metadata: Mapping[str, str | int | float | bool]   # JSON scalars only
    created_at: datetime   # timezone-aware (naive rejected)
    status: IngestStatus = PENDING
    content_sha256: str    # derived from CONTENT
    source_path: str       # relative to data_dir, '\' normalised to '/'
    page_count: int | None
    error: str | None      # required iff status == FAILED
    ingested_at: datetime | None   # required iff status == INDEXED

    # factory (validates, derives both identifiers)
    Document.create(source_path=..., filename=..., media_type=..., content=..., ...)
    # derived: .char_count .is_empty .is_supported .is_terminal .summary()
    # lifecycle: .with_status(...) .allowed_transitions()
```

**Why two identifiers.** `doc_id` hashes the *source path*; `content_sha256`
hashes the *bytes*. If `doc_id` were content-derived, editing a file would
produce a brand-new document and its stale chunks would linger in the index
forever. With path-derived ids, an edit keeps the same `doc_id` (so Stage 23
can replace old chunks) while `content_sha256` changes (so §5.2.6 detects
the edit). A rename changes `doc_id` but not the hash — that is exactly how
§5.2.6 distinguishes *renamed* from *modified*.

**Status lifecycle** (terminal states accept no further transitions):

```
PENDING ──▶ PARSING ──▶ INDEXING ──▶ INDEXED   (terminal)
   │           │            │
   └───────────┴────────────┴──────▶ FAILED    (terminal, requires error)
```

`with_status()` returns a *new* record (`frozen=True` forbids assignment),
so one worker can never mutate a record another is reading. Validation
enforces cross-field rules: `error` allowed only on `FAILED`, `ingested_at`
required on `INDEXED`.

**Validation rejects:** malformed `doc_id`, filename containing separators /
control characters / `..`, non-MIME `media_type`, naive datetimes, negative
`page_count`, absolute or traversal `source_path`, non-scalar or nested
metadata, and `status` not an `IngestStatus`. Errors are
`app.domain.errors.ValidationError` with a stable `code`.

**`summary()` deliberately omits `content` and `content_sha256`** so document
text can never reach a log line or an API list payload.

Helpers: `make_doc_id(path)`, `make_chunk_id(doc_id, index)`,
`content_sha256(bytes|str)`, `SUPPORTED_MEDIA_TYPES`,
`EXTENSION_TO_MEDIA_TYPE`.

**`Chunk`** — implemented in Stage 18, the retrievable half of a document:

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str     # f"{doc_id}:{index}", derived by .create()
    doc_id: str       # the document it came from
    index: int        # 0-based, counted across pages and sections
    content: str      # the text to embed; .summary() never includes it
    metadata: Mapping[str, str | int | float | bool]   # JSON scalars only

    # factory (derives chunk_id; the same rule Document.create follows)
    Chunk.create(doc_id=..., index=..., content=..., metadata=...)
    # derived: .char_count .summary()
```

`Chunk.create()` derives `chunk_id` rather than accepting one, so an id and
a position can never disagree — a disagreement would quietly break
idempotent re-ingestion. Validation rejects a mismatched `chunk_id`, a
malformed `doc_id`, a negative index, empty content and non-scalar
metadata, reusing the same helpers `Document` validates with. `summary()`
omits `content` for exactly the reason `Document.summary()` does. How a
document is split into chunks is §5.11.

**Not yet created** (arrive with their own stages, still planned):
`RetrievedChunk` → Stage 30, `Citation` + `Answer` → Stage 43,
`TokenUsage` → Stage 51.

### 5.2 Ingestion

Responsibility: turn a stored file into indexed, citable chunks. Idempotent per `doc_id` (content-hash IDs; re-upload overwrites rather than duplicates).

```python
# Design sketches, written before any stage ran. The sections marked
# "implemented" are authoritative where they differ; §5.11 is the chunker's
# and §5.12 the embedder's.
class DocumentLoader(Protocol):
    def load(self, path: Path, *, doc_id: str, media_type: str) -> TextDocument | PdfDocument: ...
        # originally drawn as list[LCDocument]; Stages 11-12 built native
        # records instead (§5.2.2 / §5.2.3) and the types stayed framework-free

class Chunker(Protocol):
    def split(self, document: TextDocument | PdfDocument, *, doc_id: str) -> list[Chunk]: ...
        # RecursiveCharacterSplitter(1000, 150); markdown/table-aware for structured text;
        # emits chunk_id = f"{doc_id}:{index}", preserves page/section metadata

class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
        # batched (e.g. 128), retries with exponential backoff on 429/5xx —
        # batching/retries Stage 21 (§5.14); interface Stage 19 (§5.12),
        # adapter Stage 20 (§5.13)
    def embed_query(self, text: str) -> list[float]: ...

class IngestPipeline(Protocol):
    def run(self, doc_id: str) -> IngestResult: ...
        # load → chunk → embed → upsert → registry.update_status(INDEXED | FAILED)
        # implemented Stage 23 (§5.16): run(path, source_path=...) — the
        # caller locates the file (§5.3.3's FileStore does for an upload);
        # the result is DocumentIngestResult (IngestResult is §5.2.7's
        # batch report), and update_status still belongs to whoever drives
        # the registry (§5.3.2), which closes the delete→upsert window
```

### 5.2.1 Discovery (`ingestion/discovery.py`) — implemented in Stage 10

Classifies files under a root; **never opens file contents** (loaders do that).

```python
discover(root, *, base_dir=None, recursive=True) -> DiscoveryResult

@dataclass(frozen=True, slots=True)
class DiscoveredFile:     path; relative_path; filename; media_type; size_bytes
                          # .suffix property, lower-cased
class UnsupportedFile:    relative_path; filename; suffix; reason
class DiscoveryResult:    root; files: tuple[...]; unsupported: tuple[...]
                          # .is_empty .supported_count .unsupported_count
                          # .total_size_bytes .relative_paths()

describe(result) -> dict  # JSON-safe counts, root NAME only (never abs paths)
```

**Containment (the security property).** Every returned path is checked with
`resolve().relative_to(root)`:

* `followlinks=False` in `os.walk` → directory symlinks are never descended,
  so `data/escape -> C:\Users` cannot be walked and indexed.
* Symlinked *files* are checked individually (a file link can escape even
  when directory links are not followed) → reported as unsupported with
  reason `symlink resolves outside the discovery directory`.
* `base_dir=` makes the configured `DATA_DIR` a hard boundary; passing a root
  outside it raises `ValidationError` / `discovery_root_outside_base`.
* `..` in a root is resolved *before* the containment test, so
  `data/../secrets` is caught.

**Classification.** Reuses `EXTENSION_TO_MEDIA_TYPE` from
`domain/models.py` (single source of truth, added in Stage 09). Matching is
case-insensitive (`.MD` works). Unsupported files get an actionable `reason`
naming the extension *and* listing the supported ones.

**Deliberately skipped, not reported as unsupported:** hidden entries
(`.gitkeep`, `.DS_Store`) and `IGNORED_DIRECTORY_NAMES`
(`.git`, `__pycache__`, `.venv`, `node_modules`, `.mypy_cache`,
`.ruff_cache`). A directory containing only `.gitkeep` is therefore
correctly `is_empty == True`.

**Errors:** missing root → `NotFoundError` (`discovery_root_not_found`,
message uses the basename so absolute paths never escape); root is a file →
`ValidationError` (`discovery_root_not_a_directory`).

**Determinism:** filenames are walked in sorted order, so repeated discovery
yields identical ordering — required for stable ids in Stage 23.

### 5.2.2 Text loaders (`ingestion/loaders.py`) — implemented in Stage 11

Loads `.txt` / `.md`. PDF has its own loader (§5.2.3); asking this module for
a PDF fails with an actionable message rather than decoding bytes as text.

```python
load_text(path, *, media_type=None, normalize_newlines=True) -> TextDocument
load(path, *, media_type=None) -> TextDocument          # alias the ingest service uses
can_load(media_type) -> bool
media_type_for_path(path) -> str                        # .md -> text/markdown, else text/plain
detect_newline_style(raw: bytes) -> str                 # crlf | cr | lf | mixed | none

@dataclass(frozen=True, slots=True)
class TextDocument:
    text; media_type; encoding; had_bom; original_newlines; newlines_normalized
    byte_size; source_name          # basename only, never an absolute path
    # .char_count .line_count .is_empty .summary()

TEXT_MEDIA_TYPES = frozenset({"text/plain", "text/markdown"})
```

**What it preserves.** Everything except line endings: headings, blank lines,
indented code blocks and ``` fences survive byte-for-byte (asserted by a
round-trip test). Chunking in Stage 18 depends on that structure, so a loader
that tidied text would silently damage retrieval.

**Line endings.** CRLF and lone-CR are rewritten to `\n` so Windows, macOS and
Linux files behave identically downstream. CRLF is replaced *first*, otherwise
a naive `\r -> \n` would produce blank lines. The original style is recorded in
`original_newlines` and `normalize_newlines=False` keeps the bytes as stored.

**Encoding.** Strict UTF-8. `utf-8-sig` decodes files with a byte-order mark and
strips it (an unstripped BOM becomes a stray `U+FEFF` in the first chunk), with
`had_bom` / `encoding` recording what happened. Decoding is strict on purpose:
silently substituting `U+FFFD` would put corrupt text into the vector index
where it is very hard to notice later.

**Reported failures** (all subclass `RAGError`; the new
`DocumentLoadError` maps to HTTP 422 because the request is fine, the *file*
is not):

| Condition | Error | Code |
| --- | --- | --- |
| Path missing | `NotFoundError` | `file_not_found` |
| Path is a directory | `ValidationError` | `path_is_a_directory` |
| Not a text media type | `ValidationError` | `unsupported_media_type` |
| Permission denied / OS I/O error | `DocumentLoadError` | `permission_denied` / `unreadable_file` |
| Bytes are not valid UTF-8 | `DocumentLoadError` | `encoding_error` |
| NUL bytes (binary renamed `.txt`) | `DocumentLoadError` | `binary_content` |

NUL bytes are rejected rather than decoded because a `.docx` or image given a
`.txt` name would otherwise "succeed" and be indexed as garbage.

**Empty is valid, not an error.** A zero-byte or whitespace-only file returns a
`TextDocument` with `is_empty == True`; §5.2.5 reports such files as *skipped*
rather than failing the batch.

**Out of scope:** OCR. A scanned *image* has no extractable text, and
inventing placeholder text for it would poison the index with fabricated
content. A text file that cannot be decoded raises a typed error so it can be
marked *skipped* with an honest reason. (A scanned *PDF* is a different case —
see below — because the file is perfectly readable, it simply has no text
layer.)

### 5.2.3 PDF loader (`ingestion/pdf_loader.py`) — implemented in Stage 12

Turns a PDF into page-tagged text using `pypdf`. Re-exported from
`ingestion/loaders.py` as `load_pdf` so callers have one import site.

```python
load_pdf(path, *, password=None, media_type=None) -> PdfDocument
can_load_pdf(media_type) -> bool

@dataclass(frozen=True, slots=True)
class PdfPage:
    page_number: int     # 1-based, as a citation reads
    text: str            # "" when the page has no text layer
    # .char_count .is_empty

@dataclass(frozen=True, slots=True)
class PdfDocument:
    pages: tuple[PdfPage, ...]   # always page_count entries, in order
    media_type; byte_size; encrypted; failed_pages: tuple[int, ...]; source_name
    # .page_count .text .char_count .has_text_layer .is_empty .summary()

PDF_MEDIA_TYPE = "application/pdf"
```

**Pages are kept, not flattened.** PDF is the one format here where a citation
can plausibly say "see page 7". Joining every page into one string would make
that unrecoverable, so `pages` is authoritative and `.text` is only a
convenience join. `pages[i].page_number == i + 1` always holds — even when a
page fails to extract, it keeps its slot with empty text — which is what lets
Stage 18 attach a page number to every chunk without extra bookkeeping.
Numbering is **1-based** to match how people count; the deliberate mismatch
with Python's zero-based indexing is documented and tested.

**A missing text layer is reported, not raised.** A scanned PDF is a valid
file with no text. It returns `has_text_layer == False` / `is_empty == True`
(the PDF analogue of the empty-text-file case in §5.2.2) so §5.2.5 marks it
it *skipped*. Genuine failures — corrupt bytes, or encrypted with no usable
password — still raise, because those are things a caller may need to act on.

**Per-page failure isolation.** A single malformed page does not discard the
document, but neither does it vanish: failures collect in `failed_pages`. If
*every* page fails, nothing was extracted at all, which is a real error rather
than a scan, so `invalid_pdf` is raised.

**Errors** (all `DocumentLoadError` with HTTP 422 unless noted):

| Condition | Code |
| --- | --- |
| Path missing / is a directory | `file_not_found` (`NotFoundError`) / `path_is_a_directory` (`ValidationError`) |
| Non-PDF media type requested | `unsupported_media_type` (`ValidationError`) |
| Encrypted, no password supplied | `pdf_encrypted` |
| Encrypted, password rejected | `pdf_wrong_password` |
| 0 bytes, corrupt, truncated, not really a PDF, unusable page tree, every page failed | `invalid_pdf` |

`pdf_wrong_password` is a separate code from `pdf_encrypted` so a UI can
distinguish "I cannot read this" from "that password was wrong" and re-prompt.
pypdf signals a wrong password by *returning* a falsy value rather than
raising, so the return value is checked explicitly.

**Log noise.** pypdf logs warnings such as `invalid pdf header` before
raising; each becomes a typed error carrying the real reason, so the `pypdf`
logger is raised to ERROR for the duration of a read and restored afterwards.

**Not implemented:** OCR (as above), form-field extraction, and digital
signatures. Embedded metadata itself is handled by §5.2.4. `password` exists
and is tested, but nothing supplies it yet — there is no password-management
feature.

### 5.2.4 Metadata (`ingestion/metadata.py`) — implemented in Stage 13

Turns the facts a file already carries — declared in the PDF info
dictionary, or visible in the first lines of the text — into one flat,
sanitised, storage-ready record.

```python
extract_pdf_info(reader) -> PdfInfo        # takes an OPEN reader, not a path
extract_text_info(text) -> TextInfo
extract_title_from_text(text) -> str | None
fallback_title(source_name) -> str | None
build_metadata(*, source_name, media_type, byte_size, text="",
               pdf_info=None, page_count=None) -> DocumentMetadata

sanitize_text(value, *, max_length=...) -> str | None
sanitize_keywords(value, *, limit=...) -> tuple[str, ...]
parse_pdf_datetime(value) -> datetime | None

@dataclass(frozen=True, slots=True)
class PdfInfo:      title; author; subject; keywords; creator; producer
                    created_at; modified_at          # .is_empty
class TextInfo:     heading_count; word_count; line_count; char_count
class DocumentMetadata:
    source_name; media_type; byte_size       # provenance, always present
    title; title_source; author; subject; keywords
    creator; producer; created_at; modified_at; page_count
    word_count; heading_count; char_count
    # .is_empty .to_mapping() .summary()

class TitleSource(StrEnum):  EMBEDDED | HEADING | FILENAME | NONE
```

**Embedded metadata is attacker-controlled, so it is sanitised.** A PDF's
`/Title` and `/Author` are arbitrary strings chosen by whoever produced the
file, and they flow onward into logs, API responses and the vector index.
pypdf passes control characters, NUL bytes, newlines and tabs through
untouched — a title of `"Bad\x00Title\x07"` round-trips verbatim — so every
value has control characters stripped, whitespace collapsed and length
capped (`MAX_TITLE_LENGTH` 200 for titles, `MAX_FIELD_LENGTH` 1000
otherwise, `MAX_KEYWORDS` 32). This is the same reason §5.2.2 refuses NUL
bytes in file *content*: untrusted bytes must not reach a place where they
can forge a log line or escape a terminal. Nothing is ASCII-folded —
Sinhala, CJK and emoji titles are legitimate and round-trip correctly, so
folding them to `?` would destroy real information while appearing to tidy
up.

**Absent values are omitted, never `None`.** `DocumentMetadata.to_mapping()`
drops missing fields rather than emitting them, because ChromaDB rejects
`None` and non-scalar metadata values. One mapping is therefore valid for
Chroma, for the domain `Document`, and for JSON. Datetimes are ISO-8601
strings and keywords are one comma-joined string for the same reason —
neither store accepts datetime or list values. `summary()` is the
deliberate opposite: it is built field-by-field and reports every field
with `None` for absent ones, so a log line shows what was *missing* as well
as what was found. Neither ever contains document text.

**Dates are parsed to timezone-aware UTC, or dropped.** `parse_pdf_datetime`
implements the documented `D:YYYYMMDDHHmmSSOHH'mm'` format directly rather
than calling pypdf, whose only exposed parser for it is private
(`pypdf._utils.parse_iso8824_date`) — building on a private API means a
patch release could break ingestion with no warning of ours. Every
component after the year is optional, so a bare `D:2026` is valid. A naive
input is interpreted as UTC (the domain model rejects naive datetimes,
because the same ambiguous timestamp means different moments on different
machines), and a malformed date such as `not-a-date` yields `None` rather
than raising: pypdf's own parser raises `ValueError` here, and one corrupt
field must not lose an otherwise readable document.

**Title provenance is recorded, not guessed at.** `build_metadata` resolves
the title embedded-first, then the first Markdown heading, then the
filename, and stores which one won in `title_source`. An embedded title was
declared by the author; a heading is a reasonable guess; a filename slug is
usually machine-generated and means nothing. Collapsing them would hide
which is which. Both text-derived steps inspect **only the first 20 lines**
(`_TITLE_SCAN_LINES`) — a bound found to be necessary by a test: a regex
search over the whole document happily adopted a heading 500 lines down as
the document's title, which the docstring promised would not happen. For
the same reason `is_empty` reports `False` only when the *document*
declared something: a filename-derived title is something we invented for
display, and counting it would make a metadata-free PDF claim otherwise.

**Layering.** This module imports nothing from the loaders. It consumes
primitive inputs (an open reader, a string, a byte size) and produces
values; `pdf_loader` imports *from* it to capture `PdfInfo` while its reader
is already open and decrypted, so the file is never read twice. That capture
runs after `_decrypt`, because an encrypted PDF's info dictionary is not
readable before decryption. Composing a loaded document into a full record
is the ingest service's (§5.2.5) job, which keeps the import graph acyclic.

**Errors:** none. Every failure here degrades to a missing value, which is
why no row appears in the error table. The three `except` guards around
pypdf access are tested with stubs rather than left as unexecuted
defensive code.

**Not implemented:** XMP (XML) metadata, `/Trapped` and custom info keys
(unknown keys are unbounded attacker input with no meaning to us), and
language detection.

### 5.2.5 Ingest service (`ingestion/service.py`) — implemented in Stage 14

The composition point the previous three sections were building toward:
given a directory, decide which files to read, read them, describe them,
and report honestly on every one.

```python
ingest(root, *, base_dir=None, recursive=True) -> IngestResult
ingest_file(path, *, media_type=None, relative_path=None,
            size_bytes=None) -> FileResult
describe(result) -> dict           # alias of IngestResult.summary()

class FileOutcome(StrEnum):  LOADED | SKIPPED | FAILED

@dataclass(frozen=True, slots=True)
class FileResult:
    source_name; relative_path; media_type; byte_size: int | None
    outcome: FileOutcome; code; reason
    document: TextDocument | PdfDocument | None
    metadata: DocumentMetadata | None; page_count: int | None
    # .is_loaded .is_skipped .is_failed .is_processed .text .summary()

class IngestResult:
    root; files: tuple[FileResult, ...]; unsupported: tuple[UnsupportedFile, ...]
    # .loaded_count .skipped_count .failed_count .unsupported_count
    # .processed_count .total_count .indexable_count
    # .is_empty .has_failures .relative_paths() .failed() .skipped() .summary()

LOADABLE_MEDIA_TYPES = {"text/plain", "text/markdown", "application/pdf"}
EMPTY_FILE_CODE = "empty_file";  NO_TEXT_LAYER_CODE = "no_text_layer"
```

**Outcomes are not lifecycle statuses.** Results are tagged
`FileOutcome`, deliberately a *different* vocabulary from `IngestStatus`
(§5.1). That enum describes a registered document travelling through a
pipeline, and it is owned by the registry; this service runs before any
registry exists, and its three outcomes do not map onto those five:

* A *skipped* file never becomes a document — there is no record to
  advance, so assigning it a lifecycle status would invent one.
* A *failed* file failed to be **read**, which is a different event from a
  document failing to be **indexed**. Conflating them would make a corrupt
  upload look like a vector-store outage.

So the later registry code maps these outcomes onto statuses deliberately,
rather than discovering that a skipped file cannot legally reach any
terminal state in the existing state machine.

**Only deliberate errors are caught.** The loaders already translate the
messy real world — missing files, permission errors, corrupt PDFs, rejected
passwords — into `RAGError` with a stable `code`, so catching that one type
turns a bad file into a ``failed`` result and carries on with the rest of
the batch. Anything *else* escaping is a programming bug, and bugs must not
be converted into a tidy status line: that is how a defect survives
unnoticed until production. One broken file never stops the batch; a broken
*code path* still fails loudly.

**Empty is skipped, not failed.** A text file with no content, or a scanned
PDF with no text layer, is a valid file — the loaders already report
`is_empty` rather than raising, precisely so this service can mark it
``skipped`` with an honest reason. The two cases get **different codes**
(`empty_file` vs `no_text_layer`) because they call for different actions
from the user: "you uploaded a blank file" versus "this PDF would need
OCR". Unsupported files are not even attempts: discovery classified them
before anything was opened, so they stay in `unsupported` and are never
counted among the failures.

**Media-type resolution.** `.pdf` is checked *before* falling back to
`media_type_for_path`, and that ordering matters: that function raises for
a PDF because it belongs to the text-loader module, which cannot read them.
This service reads both, so it resolves the suffix itself instead of
inheriting a guard meant for someone else's caller. Without that, a direct
`ingest_file("doc.pdf")` would raise out of the function rather than return
a result.

**Errors** (only two, and both are the *caller's* argument rather than the
file's condition):

| Condition | Behaviour | Code |
| --- | --- | --- |
| Root missing / not a directory / outside `base_dir` | raises (from discovery) | `discovery_root_not_found` / `discovery_root_not_a_directory` / `discovery_root_outside_base` |
| Caller-supplied `media_type` is not loadable | raises `ValidationError` | `unsupported_media_type` |

Every *file* problem — missing, unreadable, non-UTF-8, NUL bytes,
corrupt, encrypted, unsupported extension — returns a ``failed``
`FileResult` carrying the loader's own code instead of raising, so a
directory can be processed without a try/except per file. The asymmetry is
intentional: a media type the caller *stated* is validated eagerly (a bug
in the caller), while a media type the *file* turns out to have is reported
as a per-file outcome (a property of the data).

**Deliberately not here:** no `doc_id`, no content hash, no `Document`
record and no registry writes — those are deduplication's job (§5.2.6).
Holding them back keeps this module a pure *function of a file* (path in,
outcome out), which is what makes it testable with no store, no network and
no API key. No new dependencies: it composes modules that already exist.

### 5.2.6 Deduplication (`ingestion/deduplication.py`) — implemented in Stage 15

The service (§5.2.5) answers *"can this file be read?"*. This module answers
the harder follow-up: *"have I seen it before, and if so, in what sense?"*

```python
classify(known, *, source_path, content) -> DedupDecision

find_by_doc_id(known, doc_id) -> Document | None
find_by_hash(known, digest) -> Document | None

class ChangeKind(StrEnum):  NEW | UNCHANGED | MODIFIED | RENAMED

@dataclass(frozen=True, slots=True)
class DedupDecision:
    kind: ChangeKind
    doc_id: str                    # id the incoming path hashes to
    content_sha256: str            # hash of the incoming content
    existing_doc_id: str | None    # known record this decision concerns
    reason: str                    # one loggable line, ids only
    # .is_new .is_unchanged .is_modified .is_renamed
    # .needs_ingest .is_duplicate_content .summary()
```

**Why two identifiers, revisited.** §5.1 split `doc_id` (hash of the source
path) from `content_sha256` (hash of the content) precisely so that this
comparison is possible. Neither alone can tell the cases apart: path alone
cannot see an edit, and content alone cannot tell whether a document is new
*and* changed, or simply new. The pair gives four outcomes, each implying a
different action:

| Kind | Path known | Content known | Action |
| --- | --- | --- | --- |
| `new` | no | no | ingest it |
| `unchanged` | yes | yes | nothing to do |
| `modified` | yes | no | re-ingest, replacing its chunks |
| `renamed` | no | yes | report the document already holding it |

The two middle rows are the reason this module exists. A **modified**
document keeps its `doc_id`, so re-ingesting *replaces* its stored chunks;
mistake it for a new document and the previous version lingers in the index,
still being retrieved. A **renamed** one produces a *different* `doc_id` for
content already stored; mistake it for a new document and the same text is
indexed twice, so every retrieval returns it twice.

**Detection, not policy.** `renamed` reports a *fact* — this content already
exists under a different path — and deliberately stops there. Content alone
cannot distinguish a rename from a move from a plain copy: `spec.md` →
`api-spec.md` and `spec.md` → `backup/spec.md` produce identical hashes, yet
one should probably redirect to the existing document while the other should
be indexed independently. That call belongs to the upload API and the ingest
pipeline, which know about users and intent; this module reports what the
bytes say. So `existing_doc_id` differs from `doc_id` *only* for `renamed` —
that difference is what makes the case detectable, and for the other two
non-new kinds it points at the record being acted on.

**The path check outranks the content check.** When a document already
exists at this path, that is the decision that matters, and the content
lookup is skipped. Otherwise a file edited to equal a neighbour's content
would be reclassified as a rename, and its own stale chunks would never be
replaced. The cost is that such an edit is reported as `modified` without
mentioning the neighbour — which is fine, because re-ingesting under the
path's own `doc_id` is correct either way.

**Pure functions, registry-shaped lookups.** Nothing here opens a file,
talks to a store, or holds state: callers pass the documents they know about
and get a decision back, which is what makes this testable with no database,
no network and no API key. `find_by_doc_id` / `find_by_hash` are named to
mirror the registry Protocol (§5.4), so when a real registry lands it can
replace these linear scans with indexed queries without changing this
contract. A scan is linear — honest for what a single-process ingest sees,
and said here rather than left to be discovered.

**One rule callers must not get wrong:** hash the *same* input that built
the stored documents. `Document.create` hashes the extracted text it is
given unless a `content_hash` is passed, and a digest only ever matches what
it was computed over — comparing a file's bytes against a text-derived hash
would silently report every file as modified. The `content_hash` escape
hatch exists for exactly this (a file-level digest from discovery), and the
tests cover documents built both ways.

**Errors** (one, and again the *caller's* argument rather than the file's):

| Condition | Behaviour | Code |
| --- | --- | --- |
| `source_path` empty / absolute / contains `..` | raises `ValidationError` | `empty_source_path` / `absolute_source_path` / `path_traversal` |

A malformed path is a caller bug and propagates, matching the service's
treatment of a stated-but-wrong `media_type`: reporting it as "this file is
new" would quietly mint an id from a path that should never have reached
here. The four kinds are outcomes, not errors — there is no fifth.

**Deliberately not here:** no batch planner and no decision *store*. A
sequence of files is deduplicated by classifying each against the set as it
stands after the earlier ones were processed, which is how a sequential
ingest catches a folder containing two identical files without this module
growing any state of its own. Recording what was decided, and acting on
those decisions, is the ingest pipeline's job.

### 5.2.7 Ingest reporting (`ingestion/reporting.py`) — implemented in Stage 16

§5.2.5 answers *"what happened to this batch?"* in a form sized for **logs
and API responses**: counts, the root's name, one entry per problem file —
because a log line that lists 200 failures is noise with a cost attached. A
report is the same question asked somewhere else: once, at the end of a run,
by a person deciding what to fix.

```python
build_report(result, *, decisions=None, duration_seconds=None) -> IngestReport
format_bytes(size) -> str

@dataclass(frozen=True, slots=True)
class CodeTally:
    code: str;  count: int;  paths: tuple[str, ...]
    # .to_dict()

@dataclass(frozen=True, slots=True)
class IngestReport:
    root;  total_files;  processed_files;  indexable
    loaded;  skipped;  failed;  unsupported
    total_bytes: int | None
    failures: tuple[CodeTally, ...];  skips: tuple[CodeTally, ...]
    dedup: tuple[tuple[str, int], ...] | None;  duration_seconds: float | None
    # .is_empty .has_failures .total_problems
    # .to_dict() .to_json() .render()
```

**Derived, never a parallel record.** Every figure comes from an
`IngestResult` that already exists; nothing is re-read and no file is
opened. The report can therefore be thrown away and rebuilt after any
change, which removes the question of which of two counts is current — the
quiet failure mode of any system that keeps a summary and its source apart.

**Grouping is the whole difference.** After a 200-file ingest the useful
question is not *which files failed* but *how many failed the same way*: a
hundred `encrypted_pdf` entries are one problem with a hundred instances. So
failures and skips are bucketed by their stable `code`, ordered by frequency
with ties broken on the code itself — two runs over the same data produce
byte-identical reports. Each tally keeps its paths in discovery order,
uncapped, because a truncated list of failures is a list you cannot act on.

**Absent measurements stay absent.** `total_bytes` is `None` when no file
reported a size rather than `0`, and `duration_seconds` is `None` unless the
caller supplies it. "Never measured" and "there was nothing" are different
facts, and a report must not invent the second to fill the first.

**Deduplication tallies are optional.** Pass §5.2.6's decisions and the
report says how many files were new, unchanged, modified or renamed — the
answer to *did this run actually change anything?* Withhold them and the
field is `None`, which renders as no section at all rather than as a row of
zeros. They are supplied rather than computed because hashing content needs
bytes this module never opens; and the counts need no positional alignment
to files, so a caller may pass a subset.

**Errors:** one, and the caller's: a negative `duration_seconds` raises
`ValueError`. Only a broken clock produces one, and printing it would
launder the error into the report.

**Deliberately not here:** no document text and no absolute paths — the rule
both `summary()` methods already follow, so `render()` and `to_json()` are
safe to write to a file and commit alongside sample data. Byte formatting
uses KiB/MiB/GiB rather than KB/MB/GB: 1 KiB is 1024 bytes by definition,
whereas "MB" is read both ways in the wild and precision beats familiarity
here.

### 5.3 Storage

Responsibility: persistence + retrieval of vectors; lifecycle state of documents; storage of the uploaded source files. Two database-backed stores behind interfaces so they can be swapped (Chroma → pgvector; SQLite → Postgres), plus a plain file store over the data directory (§5.3.3).

```python
class VectorIndex(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[vector]) -> None: ...
        # implemented Stage 24 (§5.3.1); satisfies §5.16's ChunkStore
        # structurally, so the pipeline writes through to real storage
    def similarity_search(self, query: str, *, k: int,
                          filter: MetadataFilter | None = None) -> list[RetrievedChunk]: ...
        # MetadataFilter e.g. {"doc_id": {"$in": [...]}, "section": "Requirements"}
        # not yet — needs RetrievedChunk to return (§5.4), lands with retrieval
    def delete_document(self, doc_id: str) -> int: ...          # returns chunks removed
        # implemented Stage 24 (§5.3.1)
    def count(self) -> int: ...
        # implemented Stage 24 (§5.3.1)

class DocumentRegistry(Protocol):
    def create(self, doc: Document) -> None: ...
        # implemented Stage 25 (§5.3.2) — replaces rather than rejects, so a
        # finished or failed document can restart at pending as a fresh record
    def get(self, doc_id: str) -> DocumentRecord | None: ...
        # DocumentRecord is a type alias for Document (§5.3.2): status, error
        # and content_sha256 are already §5.1 fields
    def list(self, *, status: IngestStatus | None, limit: int, offset: int) -> list[DocumentRecord]: ...
        # implemented Stage 25 (§5.3.2) — limit/offset defaulted, ordered by doc_id
    def update_status(self, doc_id: str, status: IngestStatus, error: str | None = None) -> None: ...
        # implemented Stage 25 (§5.3.2) — transitions delegated to
        # Document.with_status (§5.1); raises NotFoundError for unknown ids
    def find_by_hash(self, sha256: str) -> DocumentRecord | None: ...   # dedup
        # implemented Stage 25 (§5.3.2) — indexed, deterministic "first" = lowest doc_id

class FileStore(Protocol):
    def save(self, stream: BinaryIO, *, doc_id: str) -> Path: ...
        # implemented Stage 26 (§5.3.3) — atomic replace; validates doc_id
        # before any I/O so a crafted id cannot escape the uploads directory
    def path_for(self, doc_id: str) -> Path: ...
        # implemented Stage 26 (§5.3.3) — pure, no I/O: computes where the
        # file would be; the opener of the path reports a missing one
```

`DocumentRegistry.find_by_hash` is the indexed counterpart of
`app.ingestion.deduplication.find_by_hash` (§5.2.6), which still scans a
plain sequence: the storage behind the indexed lookup has existed since
Stage 25 (§5.3.2), but nothing wires the scan to it yet. Same name, same
contract, so the swap is a change of where the lookup runs, not of what
it means.

Chroma specifics: single persistent collection `techdocs` (HNSW, cosine), `persist_directory` on a Docker volume in single-node mode or Chroma server mode if the api/worker split needs a network share. Metadata schema: `doc_id, chunk_id, filename, page, section, media_type, ingested_at`.

### 5.3.1 Chroma index write path (`storage/chroma.py`) — implemented in Stage 24

The first of §5.3's stores, and the write half of its
`VectorIndex` contract: `upsert`, `delete_document` and `count` over a
persistent collection — precisely the three methods
`IngestPipeline` (§5.16) needed to stop depending on its in-memory
double. The class satisfies §5.16's `ChunkStore` structurally (same
method names, same signatures), so the pipeline's store argument moved
from the test double to real storage with no glue code, and a test runs
the whole pipeline against a real on-disk index.

**§5.3's chroma specifics, implemented literally.** One collection
`techdocs`, HNSW with cosine distance — cosine because retrieval cares
about direction and embedding models do not promise magnitude.
`persist_directory` comes from `Settings.chroma_persist_dir` rather than
a hard-coded path, which is also how tests keep their writes inside
temporary directories. Rows are keyed by `chunk_id`; each carries the
chunk's text (retrieval quotes it back later) plus the metadata schema
`doc_id, chunk_id, filename, page, section, media_type, ingested_at` —
the first four from `filename`/`media_type`/`page`/`section`, copied
verbatim out of `Chunk.metadata` exactly as §5.11 designed the chunker
to provide them, and the rest written by the index.

**The index's three keys are written last.** `doc_id`, `chunk_id` and
`ingested_at` overwrite any same-named key a chunk might carry, because
`doc_id` is the field `delete_document` filters on: a mislabelled chunk
that could spoof it would make the overwrite protocol (§5.16) delete or
spare the wrong document's rows.

**No embeddings happen here.** Vectors arrive already aligned by
§5.15, so the collection is created without an embedding function and a
write can never trigger a model download. `similarity_search` is
deliberately absent until `RetrievedChunk` exists to return (§5.4); when
it lands, it will embed with our `Embedder` and query this same
collection — a new method, not a change to these.

**Error translation is a policy, not a blanket `try`.** Chroma speaks
`ChromaError` and its subclasses, the filesystem speaks `OSError`, the
store beneath speaks `sqlite3.Error` — all three become
`DependencyError` (`dependency_unavailable`), with the original message
kept in `detail` only. Two codes stay with the embedder family by
deliberate choice: a chunk/vector count mismatch is caught *before*
chroma's bare `ValueError` as `embedding_count_mismatch`, and chroma's
`InvalidArgumentError` — reachable through our typed API only as "a
vector disagreed with the collection's dimension" — reports
`embedding_dimension_mismatch`. Every existing holder of both codes
raises `DependencyError` too, so one alert rule still catches the same
failure class wherever it is first noticed (the §5.15 rule, applied to
a second seam). A blank collection name is rejected up front as a
`ValidationError`, so caller mistakes get a 400-shaped error rather than
whatever chroma might make of them.

### 5.3.2 Document registry (`storage/registry.py`) — implemented in Stage 25

The second of §5.3's stores: where the *lifecycle* of a document
lives — its status, its checksum, the record itself. Everything that
has been waiting on a registry now has a home. §5.2.2's service noted
it "runs before any registry exists"; §5.16's pipeline kept an
in-memory list named after this Protocol precisely so this stage could
replace it; and §5.1's status graph finally has something to apply it
to.

**SQLite, from the standard library.** No new dependency, one ordinary
file next to the rest of the configured data, and every test runs
against a temporary directory with no server to start. What §5.3
promises is the Protocol, not the engine — SQLite → Postgres stays a
swap of implementation, and the SQL is deliberately uninteresting:
one table, two indexes (on the checksum and the status), `?`
placeholders, nothing interpolated.

**The model owns the rules; the registry owns the rows.** Every status
change goes through `Document.with_status` (§5.1), so the transition
graph, the "`failed` must carry an error" rule and the `ingested_at`
stamp are enforced in exactly one place — the registry adds only the
two things the model *cannot* know: `NotFoundError` for an id it never
stored, and persistence of the result. A registry that re-implemented
the graph would be a second copy waiting to drift from the first.

**`create` replaces rather than rejects.** §5.1 makes terminal states
permanent on a *record*, with the documented remedy that "a finished
document must be re-ingested (a fresh record)". `create` is that fresh
record: re-registering a `doc_id` overwrites every column and returns
the document to `pending`, which is what lets an edited file — or a
retry after a crash — start over instead of being stuck at `indexed`
or `failed`. §6.1's 409 for a duplicate upload is decided earlier, by
§5.2.6's deduplication, never reaching this method.

**`DocumentRecord` is a name, not a second type.** The sketch's return
type is a `type` alias for `Document`, because status, error and
`content_sha256` — everything a registry manages — are already fields
of §5.1's model; a parallel record class would be a second source of
truth for the same twelve columns. Full text is stored as well, so a
round-trip is exact: every other component hands out complete
`Document` objects, and a registry returning hollowed-out copies would
be a different kind of thing everywhere it was used.

**Determinism, everywhere.** Every query orders by `doc_id` — unique
and stable, so paging never skips or repeats a row. `find_by_hash` is
the indexed counterpart of §5.2.6's sequence scan, same name and
contract on purpose: "first" is the lowest `doc_id`, so one hash
backing several paths (a renamed copy) always resolves to the same
document. Negative `limit`/`offset` are rejected before sqlite can
read `LIMIT -1` as *unbounded*.

**Proven by reopening the file.** Tests round-trip a document through
SQLite and back to a `==`-equal model instance, walk the full
lifecycle, assert terminal states refuse further moves, restart a
finished record with a fresh `create`, page and filter listings, and
translate an unopenable path into `DependencyError`. The closing test
seeds an `IngestPipeline` from `registry.list()`: an unchanged file is
skipped with zero embeds and zero store operations — §5.16's stand-in
list, now supplied by the real thing.

### 5.3.3 File store (`storage/file_store.py`) — implemented in Stage 26

The third of §5.3's stores — and the one everything else has
been talking around: *where is the file?* The ingest service records a
`source_path`, the pipeline opens whatever path it is handed, and
§6.1's worker carries nothing but a `doc_id` — somebody has to turn
that id back into bytes. `FileStore` is that somebody, in two methods:
`save` writes a stream, `path_for` says where it went.

**§6.1's layout, taken literally.** Files live at
`data/uploads/{doc_id}/original`, one directory per document as the
sequence diagram draws it, with a fixed name inside because `save`
receives only bytes and an id — the original *filename* belongs to
§5.1's record, which the registry holds, not to the store. The
directory leaves room for whatever later sits beside a source
(extracted text, thumbnails) without a layout change. The `uploads`
root comes from `Settings.data_dir`, so pointing `DATA_DIR`
somewhere else moves uploads along with everything else.

**No extension, so the media type travels with the task.** A stored
file has nothing to infer a type from, deliberately: bytes are opaque
here. §6.1's api decides the type while it validates the upload and
hands `media_type` to the worker next to the `doc_id`; a caller that
omits it falls back to `text/plain` — right for text and wrong for a
PDF. The store refuses to guess; guessing is how a PDF gets read as
plain text.

**An id becomes a path, so the id is checked first.** `doc_id` is
validated as §5.1's format — `doc_` plus 16 lowercase hex, exactly
what `make_doc_id` mints — *before* any path is built or any
directory created. `..`, separators and drive letters cannot reach
outside the uploads directory, and a rejected id costs no I/O at all.
`path_for` itself is pure: asking about an unknown id computes a
path and creates nothing, and the component that opens the path
reports the absence (`file_not_found`) rather than the lookup.

**Writes replace atomically.** Re-uploading overwrites (§5.2's
"re-upload overwrites rather than duplicates"): bytes go to a
`.part` sibling and are moved into place with `os.replace`, so a
reader sees the old file or the new one and never a half-written one.
A failed write deletes its own staging file and leaves the previous
good copy untouched; OS-level trouble arrives as `DependencyError`,
same policy as §5.3.1 and §5.3.2.

**Proven by failing, not just by writing.** Tests round-trip binary
payloads byte-for-byte, overwrite without leaving debris, reject a
parametrised gallery of malformed ids with zero I/O, translate a
blocked root and a stream that dies mid-copy into `DependencyError`
(with the old file surviving), and — the hand-off this stage exists
for — save bytes and feed `path_for`'s result straight into the real
ingest pipeline, which chunks them into the store in
`delete`-then-`upsert` order.

### 5.3.4 Storage contract tests (`tests/unit/test_storage_contract.py`) — implemented in Stage 27

The contract half of the storage vertical slice (§13, step 2), closing
out §8's testing row — *each `Protocol` implementation conforms* — with a
suite of its own.

**Three registries, one set of rules.** The contract tests run against
every implementation registered in `STORAGE_VECTOR_INDEXES`,
`STORAGE_DOCUMENT_REGISTRIES` and `STORAGE_FILE_STORES`: the Chroma
write path, the SQLite document registry and the local-disk file store
today, each joined later by adding one factory to its registry, and the
rules below running for the newcomer automatically.

The vector contract pinned here is the **write half** for now — `upsert`,
`delete_document` and `count`; the read half (`similarity_search` with its
`MetadataFilter`, returning `RetrievedChunk`) lands with retrieval
(§5.4) and joins this suite then.

**Cross-store legs close the loop.** The behaviour worth proving is not
any single store but the hand-offs between them (§6.1): an upload saved by
the file store is ingested through the real pipeline into the real vector
index, and the registry is the state that ties the two together — the
status a worker is expected to move, and the dedup counterpart a content
hash lookup reaches. Two legs are proven:

* **Happy path** — save → ingest (`created`) → index holds the chunks →
  registry marked `INDEXED` → `find_by_hash` resolves → re-ingest is a
  skip.
* **Failure isolation** — a refused embed leaves no chunks in the index,
  keeps the registry un-advanced and the stored file intact.

Both legs run against real storage (temporary Chroma directory, temporary
registry file, temporary uploads root) with the offline embedder double.

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

### 5.9 Logging (`observability/logging.py`) — implemented in Stage 16

Structured logs, and the property that matters more: **document text does
not reach them.** Extraction produces exactly the material that must not be
copied into a log — a PDF's full text, a chunk, an assembled answer — and it
is a single `logger.info("loaded", text=doc.content)` away from being
retained in a system whose memory far outlives the index it was debugging.

```python
configure_logging(*, level="INFO", allow_document_content=False,
                  renderer="json" | "console", max_value_chars=512) -> None
configure_from_settings(settings) -> None
make_content_redactor(*, allow_document_content=False, max_value_chars=...) -> Processor
get_logger(name=None) -> BoundLogger

CONTENT_KEYS               # key names treated as document text
REDACTED = "<redacted>"
DEFAULT_MAX_VALUE_CHARS = 512
```

**Three layers, because no single one is enough.**

1. **Key deny-list.** A value logged as `content`, `text` or `page_content`
   becomes `<redacted>` whatever its length — a one-line secret is still a
   secret. Matching is exact rather than a substring search, because
   `text_file_count=3` must survive: a log that lies about how many files it
   saw is worse than one showing too little. A flagged key blanks its value
   *wholesale*, structure nested inside included — the key said the value is
   content, so the value goes.
2. **Size guard.** Any string longer than `max_value_chars` becomes
   `<omitted N chars>` *whatever it is called under*. This is the backstop
   for key names the deny-list has never heard of. The default sits below
   the default chunk size (1000) so a whole chunk cannot fit, and the
   length is kept rather than a truncation preview — a preview is content
   too.
3. **Type collapsing.** `Document` is a frozen dataclass whose `repr`
   quotes its own `content`, so handing one to a logger would smuggle the
   text past the first two layers: the value is neither a flagged key nor a
   long string, just an object whose `repr` happens to contain it.
   Unrecognised objects become `<Document>`; `str`, numbers, `None`,
   `pathlib.Path`, `enum.Enum` and dates survive as themselves (a `Path` as
   a plain string, an `Enum` as its value, so `json.dumps` never has to
   fall back to `repr`).

**Redaction is on unless deliberately turned off.**
`allow_document_content=True` disables all three layers, and
`Settings.LOG_DOCUMENT_CONTENT` defaults `False`, so a fresh install gets
the safe behaviour without asking for it. The inversion is deliberate: the
risky state is the opt-in one.

**Why a processor rather than discipline.** Review catches the *known* leak
— the one a comment can anticipate. The processor catches the one nobody
anticipated, because it never reasons about intent; it only sees that a
value is large, oddly named, or an object. §5.2.5 and §5.2.7 obey the same
rule from the other direction (never *build* a payload containing text), and
the two halves meet: nothing constructs content-bearing output, and nothing
would print it if something did.

**Reconfiguration replaces rather than stacks.** Calling it twice does not
produce two pipelines, and `cache_logger_on_first_use=False` keeps that
true for loggers already created — which is what makes it possible to
reconfigure inside a test and have the test actually see the new processors.
The root logger gets a handler only when it has none, so an embedded or
under-test process keeps the handlers it already had.

**Errors:** an unknown `level` or a negative `max_value_chars` raises
`ValueError`. `Settings` validates `LOG_LEVEL` at startup, but this is also
callable directly, so the same six names are enforced rather than trusted.

**Not implemented:** no file destination or rotation, no request-id context
wiring, and no metrics (`observability/metrics.py`, Stage 51). This module
configures the library and guarantees the redaction; shipping where the API
puts its logs belongs with the API layer (§5.6).

### 5.10 Tooling: sample data and the command line (`scripts/`) — implemented in Stage 17

Three modules, plus one deliberate move between them.

**Sample data is generated, not committed.** `data/` is git-ignored, and
committing binary documents would be unreviewable, would bloat the history,
and would leave a newcomer with a corpus only one machine has. So
`samples.py` writes the corpus on demand from literals kept here, where
every byte can be read in review.

**Determinism is a requirement, not a nicety.** Document identity rests on
hashing the path and hashing the content (§5.1), so a generator that
produced a slightly different file each run would report the whole corpus
as *modified* on every re-ingest and hand it all back to the indexer. Hence
no timestamps, no random ids, and builders that are byte-stable across
runs — including the encrypted PDF, whose scheme could plausibly have been
salted. A test writes the corpus into two directories and compares the two
results byte for byte.

**Two directories, because two callers want different things.** `corpus/`
holds four readable documents and a three-page PDF with extractable text:
what a later retrieval stage can be scored against. `edge-cases/` holds a
blank file, a scan with no text layer, a password-protected PDF, a
non-UTF-8 export and an unsupported image: what proves that loaded,
skipped, failed and unsupported still behave. Separating them means an
evaluation stage never has to filter failures out of its own dataset, and a
demonstration never has to break real files by hand.

**`prepare` writes; it does not manage a directory.** Anything it does not
know about is never removed, and a file that differs is left exactly as the
user wrote it unless `--force` is given — a renamed sample in the generator
must not become a destroyed file on someone's disk.

**The exit statuses mirror the pipeline's own distinction.** `ingest`
exits `0` when nothing failed, `3` when the report printed but some files
failed to load, and `1` when it could not run at all; argparse owns `2`.
That is §5.2.5's *your data is wrong* against *your invocation is wrong*,
made machine-readable, and the report still prints on `3` because a status
with nothing behind it is one a caller cannot act on. Deliberate errors
reach the user as message, code and detail on stderr rather than as a
traceback — but only `RAGError` is caught. Anything else keeps its
traceback, because a bug must look like a bug.

**stdout is the report; stderr is everything else.** `--format json` puts
nothing but JSON on stdout so the output can be piped, while hints, errors
and logging stay on stderr. Logging is configured before any work, so
redaction applies to the CLI exactly as it does to the API (§5.9), but the
happy path emits nothing above `DEBUG`: a tool that annotates its own
successful run is noise.

**Where the PDF builders live.** `scripts/pdf_fixtures.py` holds them, and
`tests/unit/pdf_fixtures.py` re-exports them unchanged — so the three test
files importing the fixture needed no edits. Two consumers need the same
hand-written PDF syntax and neither may import the other: tooling that
imported tests would break the moment a build excluded the suite, while
tests importing the tooling would invert the direction dependencies are
meant to flow. One copy, in a package, with the arrow pointing from tests
toward tooling.

**The gate covers this package too.** `mypy app scripts` and
`--cov=app --cov=scripts` are set in `scripts/check.ps1` and
`.github/workflows/ci.yml`. New code exempt from two of the four checks
would be exactly the drift the structural and dependency guards exist to
prevent.

**Not implemented:** no `index` command (that arrives with the vector
store), no deduplication tallies in the printed report — `build_report`
already accepts `decisions`, but there is no registry to compare against
yet — and no file watching; `prepare` is run by hand.

### 5.11 Chunking (`preprocessing/chunking.py`) — implemented in Stage 18

Responsibility: turn what a loader returned into the `Chunk` records of §5.1
that get embedded, stored and cited. One class does it —
`RecursiveChunker(chunk_size=1000, chunk_overlap=150)` — wrapping LangChain's
`RecursiveCharacterTextSplitter`, pinned in `requirements.txt` since Stage 05.

**Splitting happens inside a *unit*, never across two.** A unit is the
largest span that must not be mixed with any other, and which one applies
depends on the source:

| Source | Unit | Provenance attached to every chunk |
| --- | --- | --- |
| PDF | one page (`PdfDocument.pages`) | `page` (1-based) |
| Markdown | one ATX heading section | `section` (full heading path) |
| Plain text | the whole file | — |

`section` is a **breadcrumb**, not a bare leaf title: `"Endpoints > GET
/payments/{id}"`. Leaf names repeat inside a single document, and the path
is the thing a reader can follow back through the document to check a
citation. Text before the first heading has no `section` at all.

Every chunk also carries `filename` and `media_type`, because §5.3's storage
schema lists them and attaching them here means the indexing stage copies
what it is given instead of reconstructing it from elsewhere. Keys a source
cannot honestly answer are **left out**, never `""` or `-1`: `section=""`
would read as "a section called nothing".

Two guarantees follow from units:

* **`page` is exact.** A chunk never straddles two pages, because it would
  have to name one — and naming the wrong page is worse for a citation than
  losing the overlap between those two pages is.
* **`section` is exact**, including for `# comment` lines inside fenced code
  blocks, which are not headings; splitting there would chop a script into
  pieces labelled after its own comments. Markdown sections use
  `from_language(Language.MARKDOWN)` separators (headings, fences, rules,
  blank lines); prose and extracted PDF text use the default
  blank-line-first recursion.

**A heading with no body folds into its first descendant.** A heading
whose next line is another heading — `## Endpoints` immediately followed
by `### GET /payments/{id}` — would otherwise be a section whose entire
text is the heading: a twelve-character chunk that can be retrieved and
answers nothing, while every endpoint below it is embedded with no idea
which heading it belongs to. The heading line is prepended to the child's
text instead, where the child's breadcrumb already contains the parent's
title, so text and label stay in agreement. A heading with no descendant
to fold into keeps its own section rather than losing its words.

**The one rule that may break `chunk_size`.** When a section is larger than
the limit, the splitter's first choice is the blank line under the heading,
which would strand `## Rate limits` as its own fourteen-character chunk —
retrievable, answering nothing — while the paragraph it introduces is
embedded with no idea which heading it belongs to.
`_fold_leading_heading` glues that lone heading to the first paragraph of
its section instead, re-slicing from the original text (never concatenating
the two pieces: the splitter drops the separator between them, so joining
them would invent text that appears nowhere in the document). The folded
chunk may exceed `chunk_size` by exactly its heading's length. That is a
deliberate trade — `chunk_size` is a quality target, not a ceiling anything
downstream depends on, since embeddings accept far more than 1000
characters and §5.4 counts tokens when assembling a prompt. A chunk with no
heading to glue never overshoots.

**Settings are validated in the constructor**: `chunk_size >= 1` and
`0 <= chunk_overlap < chunk_size`, as `ValidationError`s with codes
`invalid_chunk_size` / `invalid_chunk_overlap` rather than the bare
`ValueError` LangChain raises. A bad configuration then fails on the line
that set it, not partway through a document. The defaults mirror
`Settings`, and a test pins the two copies together so they cannot drift.

**Determinism is a requirement, not a nicety.** No clock, no randomness and
no per-document state on the instance: the same document always yields the
same chunks, in the same order, with the same ids — so re-ingesting an
unchanged file overwrites the same `chunk_id` values instead of appending a
second copy of every chunk. An input with no text (empty file, a scan) is
`[]`, not an error.

**Deviation from §5.2's original Protocol sketch:** `split()` takes
`TextDocument | PdfDocument`, not `list[LCDocument]`, for the same reason
the loaders do (§5.2.2 / §5.2.3). The pipeline's call site becomes
`chunker.split(loader.load(...), doc_id=...)` with no conversion step, and
LangChain's document type stays an implementation detail inside this module
rather than appearing in a signature the rest of the project must depend on.

**Not implemented:** no token-based sizing (characters stand in for tokens),
no overlap across page or section boundaries (stated above as the cost of
exact attribution), and no table parsing — a table stays whole when it fits
and otherwise breaks between rows, which is all the separator list promises.

### 5.12 Embedder contract (`embeddings/base.py`) — implemented in Stage 19

The first half of embedding generation is not a provider call; it is the
agreement that makes providers interchangeable. `Embedder` is a structural
`Protocol` with exactly the two methods §5.2 sketched: `embed_documents(texts)`
returns one vector per input, in input order, and `embed_query(text)` returns
one vector for one question. Application code asks for an `Embedder` and never
for `OpenAIEmbeddings`, so changing providers later is an import change rather
than a refactor — §1's promise, made checkable instead of aspirational. The
interface sits with its component in `embeddings/base.py` (§4's layout), not
in `domain/`, which stays data-only: models and errors, no behaviour contracts.

**The property everything downstream rests on:** a text embeds the same way
whichever method is called — `embed_query(t)` equals
`embed_documents([t])[0]`. Retrieval compares query vectors against document
vectors, so two different spaces would make every similarity score
meaningless, and the failure would present as "retrieval is badly tuned"
rather than as a mismatched embedder. The contract suite treats this as a
rule, not a coincidence.

**Shape rules, enforced rather than documented:** one vector per input text
in order; a single dimensionality shared across calls and across both
methods; every component a finite `float`, never an all-zero vector (a zero
vector has no direction and would tie with everything); an empty batch is
`[]`, not an error; and blank or non-text input raises `ValidationError` with
code `empty_embedding_input` *before* any network call. The rules live in
`tests/unit/test_embedder_contract.py`, which runs them against every
implementation in its registry — `mypy` checks each registration against the
Protocol at that same line — so a future provider joins with one added line
and cannot quietly disagree with the others.

**`FakeEmbedder` (`tests/fakes.py`) is a lexical double, honestly named.**
It SHA-256s each word into one of `dimension` buckets, counts the buckets and
L2-normalises, so shared words yield a positive cosine similarity and
retrieval tests can ask "does the right chunk win?" with no network and no
cost. It models no meaning — "car" and "automobile" are unrelated to it —
which its docstring states outright; golden tests that need real semantics
must use a real model. SHA-256 rather than Python's `hash` is deliberate:
`hash` is salted per process, so a double whose output changes between runs
would break the reproducibility every other test assumes. The double lives
under `tests/` rather than `app/` precisely because it must never be
mistaken for a provider in production code.

**Batching and retries now live behind the contract** (§5.14): the adapter
embeds texts in order-preserving batches and every call runs through one
bounded retry policy, so implementations of the contract are expected to
absorb transient provider failures rather than surface them on the first
429. Dimension *policy* stays out of the contract: it requires one
consistent dimension but pins no number, leaving `Settings.embedding_model`
in charge of what the vectors actually are.

### 5.13 OpenAI adapter (`embeddings/openai_embedder.py`) — implemented in Stage 20

The first implementation of §5.12's contract, wrapped around
`langchain_openai.OpenAIEmbeddings` (§7.3 pinned that class for exactly this,
and it already ships in the pinned dependencies). It does three jobs in call
order: validate input with the shared rule from `base.py` — a blank text is
answered locally with `empty_embedding_input` rather than paid for with a
round trip; delegate the embedding; then validate the response and translate
failures.

**Response validation exists because of what a broken value would cost
later.** The count must equal the input count, or `chunk_id` *i* silently
pairs with vector *j* and citations point at the wrong text. Components must
be finite `float`s — one `NaN` poisons every cosine it joins. Vectors in a
batch share one dimension, and once an instance has learned its dimension
from the first response, a different one is refused: a model changing
mid-run fails loudly here rather than as a mysterious rejection from the
vector-store stages much later. Violations raise `DependencyError` with
codes `embedding_invalid_response`, `embedding_count_mismatch`,
`embedding_invalid_vector` or `embedding_dimension_mismatch` — the provider
misbehaved, which is a 502-class problem, not a caller mistake.

**Provider exceptions join the hierarchy; provider messages do not.**
`openai.AuthenticationError` becomes `ProviderAuthenticationError`,
`RateLimitError` becomes `ProviderRateLimitError`, and everything else
(connection failure, timeout, 5xx, refused 400s) becomes `DependencyError`.
The translation copies only the exception class name and HTTP status into
`detail`, because OpenAI's authentication errors quote (part of) the key
they rejected, and no credential may travel through `message`/`detail` —
the rule `errors.py` states outright. The original exception rides along as
`__cause__`, so deliberate callers keep full context while logs and API
responses stay clean. A test pins this by asserting the secret never
appears anywhere in the raised error.

**Still no retries here, on purpose.** The client is built with
`max_retries=0`, so the openai transport cannot invent retry policy behind
our back; the bounded retries with exponential backoff live in
`batching.py` (§5.14), wrapping this adapter's calls, where one policy can
be seen, tested and counted instead of two silently multiplying worst-case
latency. `from_settings` assembles that client from `Settings` — model,
timeout, and the key via `require_openai_key()`, which fails with an
actionable message *before* any client exists rather than as a cryptic 401
from the API.

**Offline by construction.** The delegate is typed by a local
`_ClientLike` Protocol — langchain's shape, which takes `list[str]` where
`Embedder` promises `Sequence[str]` — so tests inject a stub needing no
inheritance, while `mypy` verifies the real class against the Protocol at
the single construction site. `StubEmbeddings` (`tests/fakes.py`) is that
stub: `FakeEmbedder`'s vectors plus knobs for corrupting output and raising
provider errors, and the adapter appears in the contract registry with it,
so the shared suite now runs against two implementations. No live call has
been made — that still needs a real `OPENAI_API_KEY`.

**Not implemented:** `dimensions` is never set, so vectors come at whatever
the model's full size is (`.env.example` warns that changing
`EMBEDDING_MODEL` invalidates the index); no token accounting beyond what
`OpenAIEmbeddings` does internally to split over-long inputs.

### 5.14 Batching and retries (`embeddings/batching.py`) — implemented in Stage 21

One embedding call can fail in two unrelated ways, and each needs a
different answer. **Too much at once** is answered by `iter_batches`:
consecutive, order-preserving slices of at most `DEFAULT_BATCH_SIZE` (128,
the sketch's "e.g. 128" in §5.2). The last batch may be shorter, empty
input yields no batches at all, and `require_batch_size` rejects sizes
below 1 with `invalid_batch_size` — a bad size fails on the line that set
it, not as a mysteriously short document mid-run. Ordering is the whole
point: vector *i* must stay paired with text *i*, or chunk ids start
citing the wrong text downstream.

**Not right now** is answered by `with_retries`, and the classification is
the interesting part. Retryable: HTTP 429, every 5xx, and connection
failures (`APIConnectionError` and the timeout beneath it — the request
may never have reached the provider, so nothing rejected it). Not
retryable: any other 4xx, because 401/403 mean the key is wrong and 400
means the request is wrong — the identical retry produces the identical
failure, it only buys quadratic latency. Also not retryable: anything
that is not an OpenAI exception at all, which would be our own bug
wearing a network's clothes; waiting two seconds does not fix a typo.
The predicate runs on the *raw* client exceptions, before §5.13's
translation, precisely because classification needs the real HTTP status.

**Bounded is a promise, not a preference.** `DEFAULT_RETRY_ATTEMPTS` is 4
(one try plus three retries); waits are exponential (1s, 2s, 4s, 8s…) capped
at 8 seconds with up to 0.5s of jitter, so a burst of callers does not retry
in lockstep. On exhaustion the *original* exception propagates (tenacity's
`reraise`), so translation still happens in §5.13 and callers still receive
`ProviderRateLimitError` or `DependencyError`. The adapter's whole
`embed_documents` call is all-or-nothing: if a middle batch gives up, the
partial prefix is discarded rather than returned with unknown chunk ids
missing. The client's own `max_retries=0` (§5.13) remains what makes this
*one* policy — three retries behind three retries would be nine requests,
and nobody could reason about the worst case.

**Retries are logged, minimally.** Each retry emits one warning carrying
the operation label, attempt number, wait in seconds, and the exception's
*type name* — never its message, which for an OpenAI auth failure quotes
the rejected key. The dependency is `tenacity` (pinned directly in
`requirements.txt` since this stage; §7.3 updated accordingly).

**Testing sleeps no seconds.** `with_retries` accepts an injected wait
strategy and the adapter accepts an injected `retry_wait`, so the retry
tests run with `wait_none()` and still prove attempt counts,
classification, exhaustion and log shape; the real exponential schedule was
exercised outside the suite by running the defaults and timing them. Batching
happens *inside* the adapter, so the contract suite (§5.12) needs no
change — its registered adapter simply passes texts through the same
batches a single delegation used to be.

### 5.15 Embed step (`embeddings/generation.py`) — implemented in Stage 22

§5.2's pipeline reads `load → chunk → embed → upsert`. Loading is
§5.2.2/§5.2.3 and splitting is §5.11; this is the third arrow:
`embed_chunks(embedder, chunks)` takes a document's chunks and returns one
vector per chunk, in the same order — precisely what the `VectorIndex`
sketch's `upsert(chunks, vectors)` pairs back. Empty input is an empty
result, not a provider call, matching the adapter's own short-circuit.

**It is a seam, so it guards the seam.** The adapter validates everything
it receives (§5.13), but its checks are per *request* while alignment is a
property of the whole document — batches are joined in this module, so it
is the only place that sees them all at once. And the contract suite
(§5.12) proves implementations behave at test time, not at 3 a.m. in
production, so the two failures that are *silent* still deserve a runtime
guard: a count mismatch reattaches citation *i* to chunk *j*'s text, and
mixed dimensions would be rejected by the store far from the cause. Both
checks are O(chunks) — counts and each vector's length — while the
expensive per-component walk (finiteness) stays where it was already paid
for. The error codes are deliberately the adapter's own
(`embedding_count_mismatch`, `embedding_invalid_vector`,
`embedding_dimension_mismatch`) so one alert rule catches the same failure
class wherever it is first noticed, and `detail` carries ids, never
content.

**Everything else passes through untouched.** A rate limit that exhausted
§5.14's retries arrives as `ProviderRateLimitError` and leaves as the same
instance, so callers handle one error hierarchy regardless of which layer
noticed first. The single info line carries `doc_id`, `chunk_count` and
`dimension` — counts only, with or without `LOG_DOCUMENT_CONTENT`.

**Proven twice over.** Unit tests drive a deliberately *disobedient*
embedder double (the contract suite keeps conforming implementations
honest; this one exists so the seam guards can be proven to fire), including
a parametrised check that no failure message ever quotes chunk content. An
integration test runs the real adapter with `batch_size=2` across five
chunks and shows the batches rejoining into one aligned set, vector for
chunk, against `FakeEmbedder`'s known output.

### 5.16 Idempotent re-ingestion (`ingestion/pipeline.py`) — implemented in Stage 23

The embedding stages can produce a document's vectors and §5.2.6 can say
whether the document needs producing at all. What was missing is the act
itself — and, specifically, the *second* act: running ingest twice must
leave the store exactly as running it once did, and re-ingesting an
edited file must leave only the new version behind. §6.1 states this as a
property of the system ("ingestion is idempotent"); this module is where
it is enforced.

`IngestPipeline.run(path, *, source_path, media_type=None)` performs
**decide → load → chunk → embed → delete → upsert → remember** and always
returns a `DocumentIngestResult` — one of four actions (`created`,
`replaced`, `skipped`, `failed`) with a stable `code` and counts. It
catches only `RAGError`; anything else escaping is a defect and is
allowed to propagate, following §5.2.2's rule.

**Decide before doing.** `classify` runs first, against the documents the
pipeline already knows. An unchanged file returns before the chunker, the
embedder or the store is touched — skipping is what makes re-ingestion
*cheap*, not merely correct. This is §5.2.6's verdict being *executed*
rather than merely reported.

**Replace, never append.** The write is `delete_document(doc_id)` then
`upsert`. Old and new chunk ids overlap (both count from `doc_id:0`), so
appending would either collide or leave a shortened document's old tail
retrievable forever — a modified file would serve stale chunks with no
visible symptom. The delete runs even for a brand-new document: an
earlier attempt may have crashed half-written, and that defensive delete
is what makes *retrying* a crash idempotent instead of duplicating.

**All fallible work first, the destructive step last.** Load, chunk and
embed happen before `delete_document`. If the provider is down and
§5.14's retries exhaust, the previous version's chunks are still in the
store — stale beats missing, and the next run simply tries again. What
remains is the narrow `delete → upsert` window: a crash there leaves the
document absent until the next run. The planned registry flow closes it
by marking a document `INDEXED` only *after* a successful upsert (§5.2's
sketch), so "not marked" is precisely what tells the next run to redo the
work.

**The known set is an in-memory registry stand-in.** `IngestPipeline`
keeps the documents it has written in a private list, seeded from the
`known` argument; lookups reuse `find_by_doc_id`/`find_by_hash` — named
after §5.3's registry Protocol on purpose, so Stages 24–27 can swap a
real `DocumentRegistry` in without changing this module's meaning. A
failed run is *not* remembered (so it retries) and an empty file is *not*
remembered (it never became a document). **The one rule seeds must
follow:** classification hashes the loaded text and `Document.create`
records the same value, so both sides agree by construction — a caller
seeding documents hashed over raw file *bytes* would make every file
read as modified.

**RENAMED is detection, not policy** (§5.2.6). The default here is
`renamed_policy="skip"`: indexing the same bytes under a second path
doubles every retrieval hit. A caller who knows the second path is a
deliberate independent copy passes `renamed_policy="ingest"` instead.

**Deviations from §5.2's sketch, stated outright.** `run` takes a
filesystem path plus an explicit `source_path` rather than only `doc_id`,
because the caller is the one holding the file — a corpus path in tests,
§5.3.3's `FileStore.path_for(doc_id)` for an upload (the explicit
identity also stops a machine-specific absolute path from silently
becoming an id — `make_doc_id` rejects those *before* the file is
opened). Its result is `DocumentIngestResult` because §5.2.7's service
already owns `IngestResult`. Document *status* stays `PENDING`:
lifecycle transitions belong to the registry (§5.3.2), applied by
whoever drives it, which is also why outcomes use this module's own
vocabulary rather than `IngestStatus`.

**Proven by running twice.** The tests ingest a real corpus file and
then re-ingest it: the second run is a skip with the store's operation
log untouched and the embedder's call count frozen; an edit to shorter
text shows `deleted_count == old count` with the old tail id gone;
reverting the edit restores byte-identical store state; a pre-seeded
embedder failure after a successful first run leaves the previous chunks
in place with *no* store operation at all — the ordering rule, asserted.
A missing file returns a `failed` result rather than raising, and both
the log line and `summary()` are checked to carry ids and counts but
never document text.

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

Failure at any stage → `status=FAILED` with `error` (e.g. `PARSE_ERROR: encrypted pdf`), other documents unaffected. Re-processing the same `doc_id` overwrites chunks (delete-by-`doc_id` then upsert) — ingestion is idempotent (§5.16).

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

### 7.0 Configuration (Stage 06 — implemented)

Configuration lives in `app/config.py` as a `pydantic-settings` `Settings`
class. Nothing else in the codebase reads `os.environ` directly.

**Source priority:** real environment variables → local `.env` file → field defaults.

```python
from app.config import get_settings
settings = get_settings()          # cached, one instance per process
settings.require_openai_key()      # raises SettingsError with a fix hint
settings.ensure_directories()      # idempotent, called once at startup
log.info("cfg", extra={"s": settings.redacted()})   # key is masked
```

| Group | Key settings | Notes |
|---|---|---|
| Provider | `OPENAI_API_KEY` (`SecretStr`), `CHAT_MODEL`, `EMBEDDING_MODEL`, `REQUEST_TIMEOUT_SECONDS`, `MAX_OUTPUT_TOKENS` | key is never serialized in the clear |
| Paths | `DATA_DIR`, `CHROMA_PERSIST_DIR`, `SAMPLE_DOCS_DIR` | sub-paths derive from `DATA_DIR` unless set explicitly; all resolved absolute |
| Chunking | `CHUNK_SIZE` (1000), `CHUNK_OVERLAP` (150) | validated `overlap < size` |
| Retrieval | `TOP_K` (5), `RETRIEVAL_THRESHOLD` (None) | threshold disabled until calibrated |
| HTTP limits | `MAX_UPLOAD_MB` (20), `MAX_QUERY_CHARS` (4000), `API_HOST`, `API_PORT` | |
| Logging | `LOG_LEVEL` (INFO), `LOG_DOCUMENT_CONTENT` (False) | content logging stays off by default |

**Secret handling.** `OPENAI_API_KEY` is a `pydantic.SecretStr`, so `repr`,
`str`, `model_dump()`, `model_dump_json()` and `redacted()` all mask it.
Verified by tests in `tests/unit/test_config.py`, including a real logger
handler round-trip. `Settings.redacted()` is the only dict intended for logs.

**Validation** runs at construction, so bad config fails at startup with a
message naming the field — not mid-request. Examples: `chunk_overlap >=
chunk_size`, an unknown `LOG_LEVEL`, `TOP_K=0`, an empty model name.

**Files:** `.env.example` (committed, placeholders only) → copy to `.env`
(git-ignored). `Settings` never requires `.env` to exist, so CI and
containerised runs can configure purely through environment variables.

### 7.1 Installed now (Stage 06 — verified importable)

| Layer | Package | Pinned version | Purpose |
|---|---|---|---|
| RAG core | `langchain` | 1.4.6 | agent/chain orchestration |
| | `langchain-core` | 1.6.11 | prompts, documents, Runnables |
| | `langchain-openai` | 1.7.2 | `ChatOpenAI`, `OpenAIEmbeddings` |
| | `langchain-text-splitters` | 1.1.3 | `RecursiveCharacterTextSplitter` |
| | `langchain-chroma` | 1.1.0 | Chroma as a LangChain vector store |
| LLM | `openai` | 3.31.0 | provider SDK |
| Vector DB | `chromadb` | 1.5.9 | persistent HNSW index |
| Parsing | `pypdf` | 6.21.1 | PDF text extraction, page by page (Stage 12); info dictionary (Stage 13) |
| Web | `fastapi` | 0.143.3 | REST API |
| | `uvicorn[standard]` | 0.54.0 | ASGI server |
| | `pydantic` | 2.14.1 | request/response validation |
| Config | `pydantic-settings` | 2.16.0 | typed `.env` settings (Stage 06) |
| Ops | `structlog` | 26.1.0 | structured logging with content withheld (Stage 16) |
| Data science | `numpy` | 2.5.4 | vector maths |
| | `pandas` | 3.0.6 | experiment tables |
| | `scikit-learn` | 1.9.1 | evaluation metrics |
| Test | `pytest` | 9.1.1 | test runner |

Environment: **Python 3.12.2**, Windows, virtualenv at `.venv/`.

`requirements.txt` pins direct dependencies with `==`.
`requirements.lock.txt` records all transitive packages for exact
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
| Docs | `python-docx`, `beautifulsoup4` | if DOCX/HTML loaders land | DOCX/HTML parsing (current loaders cover TXT/MD/PDF via `pypdf`) |
| Test | `pytest-asyncio` | 07 | async tests (`pytest-cov` already ships) |
| Quality (opt.) | `ragas`, `tiktoken`, `sentence-transformers` | 45–52 / 31 | eval, token budgeting, rerank |

LangChain 1.x moved classes that lived in `langchain` during 0.x, so all
LangChain imports stay isolated to adapter modules (`ingestion/*`,
`retrieval/*`, `generation/*`) and are asserted by
`tests/unit/test_environment.py`. Upgrades should touch one layer only.


### 7.4 Code quality standards (Stage 07 — implemented)

Configuration lives in `pyproject.toml` (one file for all tools).

| Tool | Version | Command | Purpose |
|---|---|---|---|
| Ruff (format) | 0.17.1 | `python -m ruff format .` | deterministic whitespace/import layout |
| Ruff (lint) | 0.17.1 | `python -m ruff check .` | E/W pycodestyle, F Pyflakes, I isort, UP pyupgrade, B bugbear, C4, RUF |
| mypy | 2.4.0 | `python -m mypy app scripts` | `strict = true` on `app/` and `scripts/`; looser for `tests/` |
| pytest | 9.1.1 | `python -m pytest` | `--strict-markers --strict-config`, `filterwarnings = error` |
| coverage | 7.16.2 | `python -m pytest --cov=app --cov=scripts` | branch coverage, `show_missing`; no floor — the numbers report, they do not gate |

**One command for everything (Windows):**

```powershell
.\scripts\check.ps1          # format check → lint → mypy → pytest+coverage
.\scripts\check.ps1 -Fix     # auto-format and auto-fix lint issues first
.\scripts\check.ps1 -SkipTypes   # fast inner loop while iterating
```

Dev-only packages are in `requirements-dev.txt`, **not** `requirements.txt`,
so a production install never pulls in linters. Lock file
(`requirements.lock.txt`) captures all 130 packages including dev tools.

**Error-handling convention.** Every deliberate failure raises a subclass
of `app.domain.errors.RAGError` (rooted on `RuntimeError` so Stage 06's
`SettingsError` stays compatible). Each subclass carries:

* `code` — stable machine-readable identifier (`"not_found"`), never prose
* `http_status` — what the API layer should respond with
* `detail` — developer context, **excluded from responses by default**
* `to_dict()` — JSON payload; `include_detail=True` only when logging server-side

| Class | Status | Use |
|---|---|---|
| `ConfigurationError` / `SettingsError` | 500 | bad or missing config |
| `ValidationError` | 400 (422 override) | caller supplied bad input |
| `NotFoundError` | 404 | missing document/chunk |
| `DependencyError` | 502 | OpenAI/Chroma unreachable |
| `ProviderRateLimitError` | 503 | retry later |
| `ProviderAuthenticationError` | 502 | bad provider credentials |

**CI** (`.github/workflows/ci.yml`): runs the same four checks on push/PR to
`main` under Python 3.12, with **no API key** — proving the suite is free of
paid calls. Tests marked `live` are excluded by design.

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
- Embedded metadata: PDF info-dictionary fields are attacker-controlled strings that reach logs, the API and the vector index, so every value is control-character-stripped, length-capped and scalar-only before storage (§5.2.4).
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
