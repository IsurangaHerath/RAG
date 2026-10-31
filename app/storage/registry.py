"""The document registry: lifecycle state and checksums in SQLite (Stage 25).

section 5.3 describes two independent stores: chroma holds vectors (section 5.3.1),
and this one holds what the *document* is doing — its status, its
checksum, and the record itself. Everything that has waited for a
registry now has a home: the ingest service's note that "this service
runs before any registry exists" (section 5.2.2), the pipeline's in-memory
stand-in (section 5.16, which named its lookups after this Protocol on
purpose), and section 5.1's status lifecycle, whose rules live in the model but
had nobody to *apply* them to a stored record.

Three decisions are worth spelling out.

**SQLite, from the standard library.** section 5.3 promises the pair can be
swapped (SQLite → Postgres) — what matters is the Protocol, not the
engine. sqlite3 needs no new dependency, writes to one ordinary file,
and every test runs against a temporary directory with no server to
start. The SQL is half a page: one table, two indexes, `?` placeholders
and nothing interpolated.

**The model owns the rules; the registry owns the rows.** Transitions
go through ``Document.with_status`` (section 5.1), so the graph
(``pending → parsing → indexing → indexed``, ``failed`` reachable from
anywhere, terminal states accepting nothing), the "failed needs an
error" rule and the ``ingested_at`` stamp are all enforced in exactly
one place. The registry adds only two things the model cannot know:
``NotFound`` when the ``doc_id`` is not stored, and persistence of the
result. A registry that re-implemented the graph would be a second copy
to drift.

**``create`` replaces rather than rejects.** The section 5.1 lifecycle makes
terminal states permanent *on a record*, with the documented remedy
that "a finished document must be re-ingested (a fresh record)".
``create`` is that fresh record: registering the same ``doc_id`` again
overwrites every column, which is what lets an edited file — or a
retry-after-crash — restart at ``pending`` instead of being stuck at
``indexed`` or ``failed``. section 6.1's 409 for a duplicate upload happens
*before* this method, decided by deduplication (section 5.2.6), not here.

``DocumentRecord`` is a name, not a second type
-----------------------------------------------
section 5.3's sketch returns ``DocumentRecord``; here that is a type alias for
:class:`~app.domain.models.Document` itself, because status, error and
``content_sha256`` — everything a registry manages — are already fields
of section 5.1's model. A parallel record class would be a second source of
truth for the same twelve columns. The record's full text is stored
too: every other component hands out complete ``Document`` objects, and
a registry returning hollowed-out copies would be a different kind of
thing everywhere it was used. (The file in ``FileStore`` and the chunks
in chroma remain the text that is *read*; this is the record of what
was registered.)

Determinism, everywhere
-----------------------
Every query orders by ``doc_id``: unique, stable, so paging never
skips or repeats a row, and ``find_by_hash`` — the indexed counterpart
of the sequence scan in :mod:`app.ingestion.deduplication` (section 5.2.6),
same name and contract on purpose — returns the same "first" match
every time, since one hash can back several paths (a renamed copy).

Known limits, stated up front: one connection used by the thread that
opened it (a per-thread pool arrives when the API needs one), no
schema-migration machinery yet (the DDL is ``IF NOT EXISTS`` and the
file is disposable until it is not), and a corrupt row surfaces as the
model's own ``ValidationError`` rather than a translated storage error —
we write these rows through the validated model, so anything else means
someone edited the database by hand.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from app.config import Settings
from app.domain.errors import DependencyError, NotFoundError, ValidationError
from app.domain.models import Document, IngestStatus

__all__ = ["REGISTRY_FILENAME", "DocumentRecord", "DocumentRegistry"]

#: section 5.3: the registry's file, next to the rest of the configured data.
REGISTRY_FILENAME: str = "registry.sqlite3"

#: section 5.3's return type — a persisted :class:`~app.domain.models.Document`.
type DocumentRecord = Document

_T = TypeVar("_T")


class DocumentRegistry:
    """section 5.3's ``DocumentRegistry`` over a SQLite file.

    Parameters
    ----------
    path:
        Database file to open (parents are created). Defaults to
        ``Settings.data_dir / "registry.sqlite3"``; tests always pass an
        explicit temporary path.

    Raises
    ------
    DependencyError
        The file or directory could not be opened or initialised —
        storage trouble, not caller input.

    Notes
    -----
    See the module docstring for the semantics that are not obvious from
    the signatures: ``create`` replaces, ``update_status`` delegates to
    the model's transition rules, and every listing is ordered by
    ``doc_id``.
    """

    def __init__(self, *, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else Settings().data_dir / REGISTRY_FILENAME
        self._connection = self._run("open", self._open)

    @property
    def path(self) -> Path:
        """Where this registry's database file lives."""
        return self._path

    # -- section 5.3's Protocol ----------------------------------------------------

    def create(self, doc: Document) -> None:
        """Register a document, replacing any record already under its id.

        The fresh-record entry point (section 5.1): re-registering a finished
        or failed document restarts it at ``pending`` with the new
        checksum and text, which is how re-ingestion proceeds after the
        lifecycle's terminal states.
        """
        self._run(
            "register",
            lambda: self._connection.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, filename, media_type, content,
                    metadata_json, created_at, status, content_sha256,
                    page_count, error, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path   = excluded.source_path,
                    filename      = excluded.filename,
                    media_type    = excluded.media_type,
                    content       = excluded.content,
                    metadata_json = excluded.metadata_json,
                    created_at    = excluded.created_at,
                    status        = excluded.status,
                    content_sha256 = excluded.content_sha256,
                    page_count    = excluded.page_count,
                    error         = excluded.error,
                    ingested_at   = excluded.ingested_at
                """,
                (
                    doc.doc_id,
                    doc.source_path,
                    doc.filename,
                    doc.media_type,
                    doc.content,
                    json.dumps(dict(doc.metadata), sort_keys=True, ensure_ascii=False),
                    doc.created_at.isoformat(),
                    str(doc.status),
                    doc.content_sha256,
                    doc.page_count,
                    doc.error,
                    doc.ingested_at.isoformat() if doc.ingested_at else None,
                ),
            ),
        )
        self._connection.commit()

    def get(self, doc_id: str) -> DocumentRecord | None:
        """Return the registered record, or ``None`` if it is unknown."""
        return self._run(
            "lookup",
            lambda: self._query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)),
        )

    def list(
        self,
        *,
        status: IngestStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Return records in stable ``doc_id`` order, optionally by status.

        ``limit``/``offset`` default rather than being required (the
        sketch left them bare): a default still bounds the result, while
        forcing every caller to invent a number only invites copying
        someone else's. Negative values are rejected — sqlite reads
        ``LIMIT -1`` as *unbounded*, which would turn a paging call into
        a full dump.
        """
        if limit < 0 or offset < 0:
            raise ValidationError(
                "limit and offset must be non-negative",
                code="invalid_request",
                detail=f"limit={limit} offset={offset}",
            )
        if status is None:
            sql = "SELECT * FROM documents ORDER BY doc_id LIMIT ? OFFSET ?"
            params: tuple[object, ...] = (limit, offset)
        else:
            sql = "SELECT * FROM documents WHERE status = ? ORDER BY doc_id LIMIT ? OFFSET ?"
            params = (str(status), limit, offset)
        return self._run("list", lambda: self._query_all(sql, params))

    def update_status(
        self,
        doc_id: str,
        status: IngestStatus,
        error: str | None = None,
    ) -> None:
        """Move a stored record to a new status, persisting the result.

        The transition itself is decided by ``Document.with_status``
        (section 5.1): an illegal jump raises ``invalid_status_transition``, a
        move to ``failed`` without an error raises ``missing_error``, and
        ``ingested_at`` is stamped exactly when ``indexed`` is reached.

        Raises
        ------
        NotFoundError
            No record under ``doc_id`` — status changes cannot be queued
            behind a document that was never registered.
        ValidationError
            The model rejected the transition (see above).
        """
        record = self.get(doc_id)
        if record is None:
            raise NotFoundError(
                f"document {doc_id} is not in the registry",
                code="not_found",
            )
        updated = record.with_status(status, error=error)
        self._run(
            "update",
            lambda: self._connection.execute(
                "UPDATE documents SET status = ?, error = ?, ingested_at = ? WHERE doc_id = ?",
                (
                    str(updated.status),
                    updated.error,
                    updated.ingested_at.isoformat() if updated.ingested_at else None,
                    doc_id,
                ),
            ),
        )
        self._connection.commit()

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        """Return the first record holding this content hash (section 5.2.6 dedup).

        The indexed counterpart of ``app.ingestion.deduplication.
        find_by_hash``: same name, same contract — "first" is the lowest
        ``doc_id``, so several paths carrying identical bytes (a renamed
        copy) always resolve to the same document.
        """
        return self._run(
            "lookup",
            lambda: self._query_one(
                "SELECT * FROM documents WHERE content_sha256 = ? ORDER BY doc_id LIMIT 1",
                (sha256,),
            ),
        )

    def close(self) -> None:
        """Close the connection. Idempotent, like sqlite's own close."""
        self._connection.close()

    # -- internals ----------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        """Create the directory, the connection and the schema."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self._path))
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id         TEXT PRIMARY KEY,
                source_path    TEXT NOT NULL,
                filename       TEXT NOT NULL,
                media_type     TEXT NOT NULL,
                content        TEXT NOT NULL,
                metadata_json  TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                status         TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                page_count     INTEGER,
                error          TEXT,
                ingested_at    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_documents_content_sha256
                ON documents (content_sha256);
            CREATE INDEX IF NOT EXISTS idx_documents_status
                ON documents (status);
            """
        )
        connection.commit()
        return connection

    def _query_one(self, sql: str, params: tuple[object, ...]) -> DocumentRecord | None:
        row = self._connection.execute(sql, params).fetchone()
        return self._to_document(row) if row is not None else None

    def _query_all(self, sql: str, params: tuple[object, ...]) -> builtins.list[DocumentRecord]:
        # ``builtins.`` because this class has a method called ``list``
        # (section 5.3's name), and inside the class body that shadows the type
        # in annotations — only the *annotation* needs the qualification;
        # calls would resolve to the builtin anyway.
        rows = self._connection.execute(sql, params).fetchall()
        return [self._to_document(row) for row in rows]

    @staticmethod
    def _to_document(row: sqlite3.Row) -> DocumentRecord:
        """Rebuild a validated model instance from a stored row.

        Constructed through ``Document(...)`` deliberately: the same
        ``__post_init__`` validation runs on the way *out*, so a row can
        never be handed back in a shape the model would have rejected on
        the way in.
        """
        ingested_at = row["ingested_at"]
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            media_type=row["media_type"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            status=IngestStatus(row["status"]),
            content_sha256=row["content_sha256"],
            source_path=row["source_path"],
            page_count=row["page_count"],
            error=row["error"],
            ingested_at=datetime.fromisoformat(ingested_at) if ingested_at else None,
        )

    @staticmethod
    def _run(operation: str, action: Callable[[], _T]) -> _T:
        """Run one sqlite call, translating its failures into our errors.

        Same policy as section 5.3.1's index: the engine's exceptions are
        infrastructure trouble to a caller (``DependencyError``, 502)
        with the original message kept in ``detail`` only. Our own
        ``RAGError``s — NotFound, the model's ValidationError — pass
        through untouched; they are *answers*, not failures of storage.
        """
        try:
            return action()
        except (sqlite3.Error, OSError) as exc:
            raise DependencyError(
                f"The document registry could not complete the {operation} operation.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
