"""Tests for the file store (``app/storage/file_store.py``).

Real filesystem against a temporary directory — every test writes
actual bytes. What gets pinned: exact round-trips including binary
payloads, overwrite with no debris left behind, ``path_for`` staying
pure (no I/O for any id, valid or not), the ``doc_id`` format check
that stops an id escaping the uploads directory, filesystem failures
arriving as ``DependencyError`` with any previous file intact — and,
closing section 6.1's worker leg, a file saved by the store ingesting
end-to-end through the real pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.errors import DependencyError, ValidationError
from app.domain.models import make_doc_id
from app.ingestion.pipeline import IngestPipeline, PipelineAction
from app.storage.file_store import STORED_FILENAME, UPLOADS_DIRNAME, FileStore
from tests.fakes import FakeChunkStore, FakeEmbedder

DOC_ID = "doc_0123456789abcdef"
SOURCE_PATH = "guides/api-spec.md"
MARKDOWN = (
    "# Overview\n\nThe platform exposes a payments API.\n\n"
    "# Limits\n\nBursts are smoothed per account.\n"
)


@pytest.fixture()
def store(tmp_path: Path) -> FileStore:
    """A store rooted in this test's temporary directory."""
    return FileStore(root=tmp_path / UPLOADS_DIRNAME)


def test_save_writes_the_stream_and_returns_where_it_landed(
    store: FileStore, tmp_path: Path
) -> None:
    """Bytes come back exactly — including NULs and UTF-8."""
    payload = b"\x00\xff caf\xc3\xa9 \x1b[0m\n"

    with io.BytesIO(payload) as stream:
        path = store.save(stream, doc_id=DOC_ID)

    assert path == store.path_for(DOC_ID)
    assert path == tmp_path / UPLOADS_DIRNAME / DOC_ID / STORED_FILENAME
    assert path.read_bytes() == payload


def test_path_for_is_pure_and_creates_nothing(store: FileStore) -> None:
    """An unknown id still resolves to a path — without any I/O."""
    path = store.path_for("doc_fedcba9876543210")

    assert path == store.root / "doc_fedcba9876543210" / STORED_FILENAME
    assert not store.root.exists(), "asking where a file is must not create it"


def test_save_replaces_previous_bytes_and_leaves_no_debris(
    store: FileStore,
) -> None:
    """Re-upload overwrites (section 5.2), and the staging file never lingers."""
    with io.BytesIO(b"first draft") as stream:
        store.save(stream, doc_id=DOC_ID)
    with io.BytesIO(b"final draft") as stream:
        path = store.save(stream, doc_id=DOC_ID)

    assert path.read_bytes() == b"final draft"
    assert sorted(p.name for p in path.parent.iterdir()) == [STORED_FILENAME]


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "   ",
        "..",
        "../escape",
        "guides/api-spec.md",
        "a/b",
        "a\\b",
        "doc_",
        "doc_0123456789abcde",  # 15 hex: one short
        "doc_0123456789ABCDEF",  # uppercase is not what make_doc_id mints
        "not_an_id",
    ],
    ids=lambda value: repr(value),
)
def test_ids_outside_the_document_id_format_are_rejected(store: FileStore, bad_id: str) -> None:
    """A malformed id never becomes part of a path (section 5.1's format)."""
    with pytest.raises(ValidationError) as excinfo:
        store.path_for(bad_id)
    assert excinfo.value.code == "invalid_request"

    with pytest.raises(ValidationError):
        store.save(io.BytesIO(b"x"), doc_id=bad_id)
    assert not store.root.exists(), "a rejected id must cost no I/O at all"


def test_a_non_string_id_is_rejected_rather_than_crashing(
    store: FileStore,
) -> None:
    """The isinstance guard turns a caller bug into our usual error."""
    with pytest.raises(ValidationError) as excinfo:
        store.path_for(None)  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_request"


def test_a_blocked_root_reports_dependency_trouble(tmp_path: Path) -> None:
    """A path that is a file, not a directory, is storage trouble (502)."""
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied", encoding="utf-8")
    store = FileStore(root=blocked)

    with pytest.raises(DependencyError) as excinfo:
        store.save(io.BytesIO(b"bytes"), doc_id=DOC_ID)

    assert excinfo.value.code == "dependency_unavailable"
    # Writing below a file fails at the OS level, but the errno differs:
    # Windows reports FileExistsError, POSIX reports NotADirectoryError.
    detail = excinfo.value.detail or ""
    assert any(name in detail for name in ("FileExistsError", "NotADirectoryError"))
    assert blocked.read_text(encoding="utf-8") == "occupied", "the blocker survives"


class DyingStream:
    """A binary stream that dies on its second read, mid-copy.

    Duck-typed as ``BinaryIO`` on purpose: only ``read`` is called, and
    the point is to fail *after* the staging file already exists.
    """

    def __init__(self) -> None:
        self.reads = 0

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return b"partial content"
        raise OSError("device reported an I/O error")


def test_a_stream_that_dies_mid_copy_leaves_no_partial_file(
    store: FileStore,
) -> None:
    """A failed write cleans up after itself and keeps the store consistent."""
    with pytest.raises(DependencyError) as excinfo:
        store.save(DyingStream(), doc_id=DOC_ID)  # type: ignore[arg-type]

    assert excinfo.value.code == "dependency_unavailable"
    assert not (store.root / DOC_ID / STORED_FILENAME).exists()
    assert list((store.root / DOC_ID).iterdir()) == [], "no .part debris"


def test_the_default_root_sits_under_the_configured_data_dir() -> None:
    """section 6.1's ``data/uploads`` comes from Settings, not a hard-coded path."""
    assert FileStore().root == Settings().data_dir / UPLOADS_DIRNAME


def test_a_saved_file_ingests_end_to_end(tmp_path: Path) -> None:
    """section 6.1's worker leg: ``doc_id`` → stored bytes → chunks.

    The api saves the upload, the worker gets only the id, resolves it
    through ``path_for`` and hands that path to the pipeline — the
    hand-off this stage exists to make possible.
    """
    store = FileStore(root=tmp_path / UPLOADS_DIRNAME)
    doc_id = make_doc_id(SOURCE_PATH)
    with io.BytesIO(MARKDOWN.encode("utf-8")) as stream:
        path = store.save(stream, doc_id=doc_id)

    chunk_store = FakeChunkStore()
    pipeline = IngestPipeline(embedder=FakeEmbedder(), store=chunk_store)

    result = pipeline.run(path, source_path=SOURCE_PATH, media_type="text/markdown")

    assert result.doc_id == doc_id, "the id survives the round-trip through disk"
    assert result.action is PipelineAction.CREATED
    assert result.chunk_count >= 2, "headings should split the file into sections"
    assert [op for op, _ in chunk_store.ops] == ["delete", "upsert"], (
        "the pipeline clears the doc_id first, then writes the new chunks"
    )
    assert chunk_store.ops[-1][1] == doc_id
