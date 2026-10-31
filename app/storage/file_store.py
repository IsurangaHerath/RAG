"""The file store: uploaded source bytes on disk (Stage 26).

section 5.3's last Protocol answers a question every other component has been
working around: *where is the file?* The ingest service records a
``source_path``, the pipeline opens whatever path it is handed, and
section 6.1's worker carries nothing but a ``doc_id`` — somebody has to turn
that id back into bytes. ``FileStore`` is that somebody: two methods,
``save`` and ``path_for``, and one directory per document under the
configured data root.

Three decisions are worth spelling out.

**A directory per document, with one fixed file inside.** section 6.1 draws
the upload as ``data/uploads/{doc_id}/…``; the ``…`` is a file called
``original``, because ``save`` receives only bytes and an id — the
original *filename* lives in the registry record (section 5.1's ``filename``
field), not here. The directory also leaves room for whatever later
wants to sit beside a source (extracted text, thumbnails) without a
layout change.

**No extension, so the media type travels with the task.** A stored
file is ``…/doc_abc123/original`` — nothing to infer a type from, on
purpose: bytes are opaque here. section 6.1's api decides the type while it
validates the upload, so it hands ``media_type`` to the worker along
with the ``doc_id``; a caller that omits it gets
:func:`app.ingestion.loaders.media_type_for_path`'s ``text/plain``
default, which is right for text and wrong for a PDF. The store
refuses to guess.

**Writes replace atomically, never tear.** Re-uploading the same id
overwrites the stored file — section 5.2's "re-upload overwrites rather than
duplicates". Bytes go to a ``.part`` sibling and are moved into place
with ``os.replace``, so a concurrent reader sees the old file or the
new one and never a half-written one; a failed write leaves the
previous good file behind and cleans up its own debris.

``doc_id`` becomes part of a path
---------------------------------
So it is checked as section 5.1's format — ``doc_`` plus 16 lowercase hex
characters, exactly what :func:`~app.domain.models.make_doc_id`
mints — before any path is built. An id carrying ``..``, a path
separator or a drive letter would otherwise reach outside the uploads
directory; the format check rejects traversal, blanks and typos with
the same ``invalid_request`` every other caller-facing validation
uses.

``path_for`` is pure
--------------------
It computes where the file *would* be and does no I/O: asking about
an unknown id costs nothing and creates nothing, and a missing file is
reported by whoever eventually opens the path (the loader's
``file_not_found``), not by the lookup itself.

Known limits, stated up front: one ``.part`` name per document, so
two concurrent saves of the *same* id would collide — acceptable
while the architecture keeps a single writer (Section11); no ``fsync``, so a
machine that loses power mid-write can lose the new file (never a
torn one); and empty bytes are stored like any other — blankness is
upload validation's policy (section 6.1), not the store's.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Final, TypeVar

from app.config import Settings
from app.domain.errors import DependencyError, ValidationError

__all__ = ["STORED_FILENAME", "UPLOADS_DIRNAME", "FileStore"]

#: section 6.1: uploads live under the configured data root.
UPLOADS_DIRNAME: Final = "uploads"

#: The single file kept per document (see the module docstring).
STORED_FILENAME: Final = "original"

#: section 5.1's ``doc_id`` — what :func:`~app.domain.models.make_doc_id` mints.
_DOC_ID_PATTERN: Final = re.compile(r"doc_[0-9a-f]{16}\Z")

_T = TypeVar("_T")


class FileStore:
    """section 5.3's ``FileStore``: save bytes under a ``doc_id``, find them again.

    Parameters
    ----------
    root:
        Directory holding one sub-directory per document. Defaults to
        ``Settings.data_dir / "uploads"``; tests always pass an
        explicit temporary path. Nothing is created until the first
        successful :meth:`save`.

    Raises
    ------
    ValidationError
        A ``doc_id`` that is not section 5.1's format (from either method).
    DependencyError
        The filesystem refused the write — storage trouble, not
        caller input.

    Notes
    -----
    See the module docstring for the semantics the signatures cannot
    say: writes are atomic replaces, ``path_for`` performs no I/O, and
    the stored name carries no extension, so the media type travels
    with the task rather than being inferred from it.
    """

    def __init__(self, *, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else Settings().data_dir / UPLOADS_DIRNAME

    @property
    def root(self) -> Path:
        """Where documents are stored — one sub-directory per ``doc_id``."""
        return self._root

    # -- section 5.3's Protocol ----------------------------------------------------

    def save(self, stream: BinaryIO, *, doc_id: str) -> Path:
        """Write the stream's bytes to this document's file and return the path.

        The stream is read from its *current* position to EOF and the
        bytes are stored verbatim: no encoding, no size policy, no
        blankness check — those belong to section 6.1's upload validation, not
        to storage. Re-saving the same ``doc_id`` overwrites atomically
        (``.part`` sibling plus ``os.replace``), so a reader never sees
        a half-written file.

        Raises
        ------
        ValidationError
            ``doc_id`` is not section 5.1's format. Raised before any I/O, so
            a bad id cannot create so much as a directory.
        DependencyError
            The directory could not be created, the file opened, or the
            replacement moved into place; the original OS error is kept
            in ``detail``.
        """
        target = self.path_for(doc_id)  # validate first: a bad id costs no I/O

        def write() -> Path:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(target.name + ".part")
            try:
                with staging.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
                os.replace(staging, target)
            except OSError:
                # Never leave our own debris behind; suppress so a
                # cleanup failure cannot mask the original error.
                with contextlib.suppress(OSError):
                    staging.unlink(missing_ok=True)
                raise
            return target

        return self._run("save", write)

    def path_for(self, doc_id: str) -> Path:
        """Return where ``doc_id``'s file lives — pure, no I/O performed.

        An unknown id still returns a path: the component that opens it
        reports the absence (``file_not_found``), and asking must not
        create directories or raise storage errors of its own.
        """
        if not isinstance(doc_id, str) or not _DOC_ID_PATTERN.match(doc_id):
            raise ValidationError(
                "doc_id must be 'doc_' followed by 16 lowercase hex characters",
                code="invalid_request",
                detail=f"doc_id={str(doc_id)[:64]!r}",
            )
        return self._root / doc_id / STORED_FILENAME

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _run(operation: str, action: Callable[[], _T]) -> _T:
        """Run one filesystem call, translating its failures into our errors.

        Same policy as section 5.3.1's index and section 5.3.2's registry: OS-level
        failures are infrastructure trouble to a caller
        (``DependencyError``, 502) with the original message kept in
        ``detail``. Our own :class:`~app.domain.errors.ValidationError`
        is raised outside this wrapper, so it passes through untouched:
        it is an *answer*, not a failure of storage.
        """
        try:
            return action()
        except OSError as exc:
            raise DependencyError(
                f"The file store could not complete the {operation} operation.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
