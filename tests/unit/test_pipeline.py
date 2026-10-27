"""Tests for idempotent re-ingestion (``app/ingestion/pipeline.py``).

The module exists to make *re-running* safe, so most of these tests run
twice (or three times) and assert the final store state — no duplicates,
no stale tails, no work repeated for an unchanged file. The failure tests
pin the ordering rule: every fallible step happens before the destructive
one, so a broken provider leaves the previous version intact.

Everything is offline: an in-memory store, a counting embedder double,
real loaders and chunker over files in ``tmp_path``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from app.domain.errors import (
    DependencyError,
    ProviderRateLimitError,
    ValidationError,
)
from app.domain.models import Document, content_sha256, make_doc_id
from app.ingestion.pipeline import (
    NO_CHUNKS_CODE,
    RENAMED_CODE,
    UNCHANGED_CODE,
    IngestPipeline,
    PipelineAction,
)
from app.ingestion.service import EMPTY_FILE_CODE
from app.preprocessing.chunking import RecursiveChunker
from tests.fakes import FakeChunkStore, FakeEmbedder

SOURCE_A = "guides/api-spec.md"
SOURCE_B = "guides/spec-copy.md"
MARKER = "CONFIDENTIAL-REINGEST-ENTRY"
SHORT_TEXT = f"{MARKER} applies per account."
LONG_TEXT = " ".join(
    f"Rate limit paragraph number {i} explains retry budgets and backoff." for i in range(12)
)


class CountingEmbedder:
    """``FakeEmbedder`` that counts calls and can be made to fail."""

    def __init__(self) -> None:
        self._fake = FakeEmbedder()
        self.calls = 0
        self.error: BaseException | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._fake.embed_query(text)


class CountingChunker:
    """Real chunker that counts splits, optionally with non-default sizes."""

    def __init__(self, inner: RecursiveChunker | None = None) -> None:
        self._inner = inner if inner is not None else RecursiveChunker()
        self.calls = 0

    def split(self, document: object, *, doc_id: str) -> list[object]:
        self.calls += 1
        return self._inner.split(document, doc_id=doc_id)  # type: ignore[arg-type]


class EmptyChunker:
    """A chunker whose documents never yield anything (no_chunks path)."""

    def split(self, document: object, *, doc_id: str) -> list[object]:
        return []


def write(tmp_path: Path, name: str, text: str) -> Path:
    """Create a corpus file under a fresh tmp directory."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def make_pipeline(**kwargs: object) -> tuple[IngestPipeline, FakeChunkStore, CountingEmbedder]:
    """Build the pipeline with fresh doubles, returning them for inspection."""
    store = kwargs.pop("store", None) or FakeChunkStore()
    embedder = kwargs.pop("embedder", None) or CountingEmbedder()
    chunker = kwargs.pop("chunker", None) or CountingChunker()
    pipeline = IngestPipeline(
        embedder=embedder,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        chunker=chunker,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    return pipeline, store, embedder  # type: ignore[return-value]


# --- Creating: what a first successful run must produce --------------------


def test_a_new_file_is_created_and_written(tmp_path: Path) -> None:
    """The happy path: decide new, chunk, embed, delete, upsert, remember."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    chunker = CountingChunker(RecursiveChunker(chunk_size=60, chunk_overlap=0))
    pipeline, store, embedder = make_pipeline(chunker=chunker)

    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.CREATED
    assert result.kind is not None and result.kind.value == "new"
    assert result.chunk_count >= 2, "the long text should split into several chunks"
    assert result.deleted_count == 0
    assert store.count(make_doc_id(SOURCE_A)) == result.chunk_count
    assert embedder.calls == 1
    assert result.document is not None
    assert len(pipeline.known) == 1


def test_write_order_is_delete_then_upsert(tmp_path: Path) -> None:
    """The overwrite protocol, visible in the store's operation log."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, store, _ = make_pipeline()

    pipeline.run(path, source_path=SOURCE_A)

    doc_id = make_doc_id(SOURCE_A)
    assert store.ops == [("delete", doc_id), ("upsert", doc_id)]


def test_chunk_ids_follow_the_stable_pattern(tmp_path: Path) -> None:
    """Ids are ``doc_id:index`` from zero — what overwrites latch onto."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, store, _ = make_pipeline()

    result = pipeline.run(path, source_path=SOURCE_A)

    doc_id = make_doc_id(SOURCE_A)
    expected = tuple(f"{doc_id}:{i}" for i in range(result.chunk_count))
    assert store.chunk_ids(doc_id) == expected


def test_summary_never_leaks_text_or_absolute_paths(tmp_path: Path) -> None:
    """Reports point at ids and counts; document bodies stay out."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, _, _ = make_pipeline()

    result = pipeline.run(path, source_path=SOURCE_A)
    rendered = str(result.summary())

    assert MARKER not in rendered
    assert str(tmp_path) not in rendered


# --- Re-running: the whole point -------------------------------------------


def test_running_twice_writes_once_and_skips_the_second_time(tmp_path: Path) -> None:
    """Idempotency proper: no second embed, no store op, no duplicate."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, store, embedder = make_pipeline()

    first = pipeline.run(path, source_path=SOURCE_A)
    state_after_first = copy.deepcopy(store.data)
    ops_after_first = list(store.ops)

    second = pipeline.run(path, source_path=SOURCE_A)

    assert second.action is PipelineAction.SKIPPED
    assert second.code == UNCHANGED_CODE
    assert first.is_created and not first.is_skipped
    assert second.is_skipped and not second.is_failed
    assert embedder.calls == 1, "an unchanged file must not be re-embedded"
    assert store.ops == ops_after_first, "an unchanged file must not touch the store"
    assert store.data == state_after_first
    assert len(pipeline.known) == 1
    assert first.document is not None


def test_an_edit_replaces_the_old_chunks_including_the_tail(tmp_path: Path) -> None:
    """MODIFIED keeps doc_id, so delete-then-upsert must remove the tail."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    chunker = CountingChunker(RecursiveChunker(chunk_size=60, chunk_overlap=0))
    pipeline, store, _ = make_pipeline(chunker=chunker)
    doc_id = make_doc_id(SOURCE_A)

    first = pipeline.run(path, source_path=SOURCE_A)
    assert first.chunk_count >= 4, "long text should produce a healthy tail to lose"

    path.write_text(SHORT_TEXT, encoding="utf-8")
    second = pipeline.run(path, source_path=SOURCE_A)

    assert second.action is PipelineAction.REPLACED
    assert second.is_replaced and not second.is_created
    assert second.kind is not None and second.kind.value == "modified"
    assert second.deleted_count == first.chunk_count, "every old chunk must be removed"
    assert store.count(doc_id) == second.chunk_count
    assert second.chunk_count < first.chunk_count
    old_tail = f"{doc_id}:{first.chunk_count - 1}"
    assert old_tail not in store.chunk_ids(doc_id), "the old tail would still be retrievable"
    assert len(pipeline.known) == 1, "the edit updates the record in place"
    assert pipeline.known[0].content_sha256 == content_sha256(SHORT_TEXT)


def test_reverting_an_edit_restores_the_original_state_exactly(tmp_path: Path) -> None:
    """A -> A' -> A must land on precisely A's store state, not a merge."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    chunker = CountingChunker(RecursiveChunker(chunk_size=60, chunk_overlap=0))
    pipeline, store, _ = make_pipeline(chunker=chunker)

    pipeline.run(path, source_path=SOURCE_A)
    original = copy.deepcopy(store.data)

    path.write_text(SHORT_TEXT, encoding="utf-8")
    pipeline.run(path, source_path=SOURCE_A)
    path.write_text(LONG_TEXT, encoding="utf-8")
    reverted = pipeline.run(path, source_path=SOURCE_A)

    assert reverted.action is PipelineAction.REPLACED
    assert store.data == original


def test_a_seeded_known_set_makes_the_first_run_skip(tmp_path: Path) -> None:
    """What a future registry supplies: already-indexed docs skip at once."""
    seed = Document.create(
        source_path=SOURCE_A,
        filename="api-spec.md",
        media_type="text/markdown",
        content=LONG_TEXT,
    )
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, store, embedder = make_pipeline(known=[seed])

    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.SKIPPED
    assert result.code == UNCHANGED_CODE
    assert embedder.calls == 0
    assert store.ops == []


def test_leftovers_from_a_crashed_attempt_are_cleaned(tmp_path: Path) -> None:
    """The defensive delete: even a 'new' write removes partial debris."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, store, _ = make_pipeline()
    doc_id = make_doc_id(SOURCE_A)
    store.data[doc_id] = {f"{doc_id}:99": [0.0] * 4}

    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.CREATED
    assert result.deleted_count == 1
    assert f"{doc_id}:99" not in store.chunk_ids(doc_id)
    assert store.count(doc_id) == result.chunk_count


# --- Failure: fallible work happens before the destructive step ------------


def test_an_embed_failure_leaves_the_previous_version_intact(tmp_path: Path) -> None:
    """Provider down after a successful first run: stale beats missing."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, store, embedder = make_pipeline()
    pipeline.run(path, source_path=SOURCE_A)
    previous = copy.deepcopy(store.data)
    ops_before = list(store.ops)

    path.write_text("Edited while the provider is down.", encoding="utf-8")
    embedder.error = ProviderRateLimitError("still rate limited")
    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.FAILED
    assert result.is_failed and not result.is_skipped
    assert result.code == "provider_rate_limited"
    assert result.chunk_count == 0, "nothing was written"
    assert result.deleted_count == 0, "nothing was deleted either"
    assert store.data == previous, "the old chunks must survive a failed embed"
    assert store.ops == ops_before, "the store was never touched on this run"
    assert len(pipeline.known) == 1, "the old record stays, so the edit retries"


def test_an_upsert_failure_is_reported_and_the_run_can_be_retried(
    tmp_path: Path,
) -> None:
    """A failed run is not remembered, so the next run does the work."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    store = FakeChunkStore()
    pipeline, store, _ = make_pipeline(store=store)
    store.fail_upsert_with = DependencyError("store is down")

    failed = pipeline.run(path, source_path=SOURCE_A)

    assert failed.action is PipelineAction.FAILED
    assert failed.code == "dependency_unavailable"
    assert pipeline.known == (), "nothing may be remembered as indexed"

    store.fail_upsert_with = None
    retried = pipeline.run(path, source_path=SOURCE_A)

    assert retried.action is PipelineAction.CREATED
    assert store.count(make_doc_id(SOURCE_A)) == retried.chunk_count
    assert len(pipeline.known) == 1


def test_a_load_failure_becomes_a_result_not_an_exception(tmp_path: Path) -> None:
    """Missing file: report it like the service does, with a stable code."""
    pipeline, store, embedder = make_pipeline()

    result = pipeline.run(tmp_path / "missing.md", source_path="guides/missing.md")

    assert result.action is PipelineAction.FAILED
    assert result.code is not None and result.reason is not None
    assert result.kind is None, "classification never ran"
    # A pre-classification failure is the one summary() with no kind:
    # it must still carry the machine code and the human reason.
    rendered = result.summary()
    assert rendered["code"] == result.code
    assert rendered["reason"] == result.reason
    assert "kind" not in rendered
    assert store.ops == []
    assert embedder.calls == 0
    assert pipeline.known == ()


def test_unexpected_errors_are_not_swallowed(tmp_path: Path) -> None:
    """A defect inside a collaborator fails loudly, per the service's rule."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    store = FakeChunkStore()
    pipeline, store, _ = make_pipeline(store=store)
    store.fail_upsert_with = TypeError("a bug, not a provider problem")

    with pytest.raises(TypeError):
        pipeline.run(path, source_path=SOURCE_A)


# --- Skips: nothing written, for four documented reasons --------------------


def test_an_empty_file_skips_with_the_service_code(tmp_path: Path) -> None:
    """A blank file is valid and unindexable — same code the service uses."""
    path = write(tmp_path, "empty.md", "")
    pipeline, store, embedder = make_pipeline()

    result = pipeline.run(path, source_path="guides/empty.md")

    assert result.action is PipelineAction.SKIPPED
    assert result.code == EMPTY_FILE_CODE
    assert store.ops == []
    assert embedder.calls == 0
    assert pipeline.known == (), "it never became a document"


def test_a_chunker_with_nothing_to_offer_skips(tmp_path: Path) -> None:
    """Loaded but unsplittable: skip with the module's own stable code."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, store, embedder = make_pipeline(chunker=EmptyChunker())

    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.SKIPPED
    assert result.code == NO_CHUNKS_CODE
    assert store.ops == []
    assert embedder.calls == 0
    assert pipeline.known == ()


def test_identical_content_at_a_new_path_skips_by_default(tmp_path: Path) -> None:
    """RENAMED is reported, not indexed — same bytes twice doubles retrieval."""
    write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, store, embedder = make_pipeline()
    pipeline.run(tmp_path / "api-spec.md", source_path=SOURCE_A)
    embeds_after_first = embedder.calls

    copy_path = write(tmp_path, "spec-copy.md", SHORT_TEXT)
    result = pipeline.run(copy_path, source_path=SOURCE_B)

    assert result.action is PipelineAction.SKIPPED
    assert result.code == RENAMED_CODE
    assert result.kind is not None and result.kind.value == "renamed"
    assert embedder.calls == embeds_after_first
    assert store.count(make_doc_id(SOURCE_B)) == 0


def test_renamed_content_is_indexed_when_the_policy_allows_it(tmp_path: Path) -> None:
    """The documented override: a deliberate copy may be indexed anyway."""
    write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, store, _ = make_pipeline(renamed_policy="ingest")
    pipeline.run(tmp_path / "api-spec.md", source_path=SOURCE_A)

    copy_path = write(tmp_path, "spec-copy.md", SHORT_TEXT)
    result = pipeline.run(copy_path, source_path=SOURCE_B)

    assert result.action is PipelineAction.CREATED
    assert result.kind is not None and result.kind.value == "renamed"
    assert store.count(make_doc_id(SOURCE_A)) == 1
    assert store.count(make_doc_id(SOURCE_B)) == 1, "both documents live, on purpose"


def test_the_default_chunker_is_used_when_none_is_supplied(tmp_path: Path) -> None:
    """Construction without a chunker still ingests, via the real default."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    embedder = CountingEmbedder()
    pipeline = IngestPipeline(embedder=embedder, store=FakeChunkStore())  # type: ignore[arg-type]

    result = pipeline.run(path, source_path=SOURCE_A)

    assert result.action is PipelineAction.CREATED
    assert result.chunk_count >= 1
    assert embedder.calls == 1


# --- Caller mistakes raise before any file is opened -----------------------


def test_an_absolute_path_without_an_identity_raises(tmp_path: Path) -> None:
    """Identity must not be a machine-specific path, even for a real file."""
    path = write(tmp_path, "api-spec.md", SHORT_TEXT)
    pipeline, _, _ = make_pipeline()

    with pytest.raises(ValidationError):
        pipeline.run(path)


def test_a_malformed_identity_raises_before_reading_anything(tmp_path: Path) -> None:
    """``..`` in source_path is rejected even when the file does not exist."""
    pipeline, store, _ = make_pipeline()

    with pytest.raises(ValidationError):
        pipeline.run(tmp_path / "gone.md", source_path="../escape.md")
    assert store.ops == []


def test_an_invalid_renamed_policy_is_rejected_at_construction() -> None:
    """A typo in our own constructor must not become quiet misbehaviour."""
    with pytest.raises(ValueError):
        IngestPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            store=FakeChunkStore(),  # type: ignore[arg-type]
            renamed_policy="overwrite",  # type: ignore[arg-type]
        )


# --- Logging: counts only ---------------------------------------------------


def test_the_success_log_carries_counts_and_never_text(tmp_path: Path) -> None:
    """One info line per write: doc_id, action, counts — no content."""
    path = write(tmp_path, "api-spec.md", LONG_TEXT)
    pipeline, _, _ = make_pipeline()

    with capture_logs() as captured:
        pipeline.run(path, source_path=SOURCE_A)

    ingested = [entry for entry in captured if entry["event"] == "document ingested"]
    assert len(ingested) == 1
    entry = ingested[0]
    assert entry["doc_id"] == make_doc_id(SOURCE_A)
    assert entry["action"] == "created"
    assert isinstance(entry["chunk_count"], int) and entry["chunk_count"] >= 1
    assert all(MARKER not in str(value) for value in entry.values())
