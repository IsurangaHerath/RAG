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
│   ├── ingestion/           # Stages 10–16
│   │   ├── discovery.py     # find supported files, block path traversal
│   │   ├── loaders.py       # Stage 11: TXT/MD -> TextDocument; re-exports load_pdf
│   │   ├── pdf_loader.py    # Stage 12: PDF -> per-page text via pypdf
│   │   ├── metadata.py      # Stage 13: sanitised PDF info + derived metadata
│   │   ├── service.py       # Stage 14: orchestrates discovery → load → metadata
│   │   ├── deduplication.py # Stage 15: stable IDs + content hashes
│   │   └── reporting.py     # Stage 16: structured per-batch ingest report
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

**Not yet created** (arrive with their own stages, still planned):
`Chunk` → Stage 18, `RetrievedChunk` → Stage 30, `Citation` + `Answer` →
Stage 43, `TokenUsage` → Stage 51. `make_chunk_id()` already exists so
Stage 18 can use the agreed id format immediately.

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

`DocumentRegistry.find_by_hash` is the indexed counterpart of
`app.ingestion.deduplication.find_by_hash` (§5.2.6), which today scans a
plain sequence for the same reason everything else here is unimplemented:
the storage for it does not exist yet. Same name, same contract, so the swap
is a change of where the lookup runs, not of what it means.

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
| Docs | `pypdf`, `python-docx`, `beautifulsoup4` | 11–12 | PDF/DOCX/HTML parsing |
| Ops | `tenacity` | 21 | bounded retries |
| Test | `pytest-cov`, `pytest-asyncio` | 07 | coverage, async tests |
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
