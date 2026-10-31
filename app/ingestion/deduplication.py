"""Content-aware deduplication: deciding whether a file is actually new.

Every document carries **two** identifiers, and the pair is what makes
deduplication decidable (see :mod:`app.domain.models`):

* ``doc_id`` hashes the **source path** — it answers *"which document is
  this?"*
* ``content_sha256`` hashes the **content** — it answers *"have I seen these
  bytes before?"*

Comparing an incoming file's pair against the documents already known gives
four outcomes, and each implies a different action:

* **NEW** — neither the path nor the content is known: ingest it.
* **UNCHANGED** — both already known: this exact document is indexed, so do
  nothing.
* **MODIFIED** — the path is known but the content changed: re-ingest, which
  *replaces* the chunks already stored under that ``doc_id``.
* **RENAMED** — the content is known at a different path: report the
  document that already holds it.

That list is the entire module. The reason it is worth isolating is that
two of these are otherwise easy to confuse, with expensive results:

* A **modified** document keeps its ``doc_id``, so re-ingesting replaces the
  old chunks. Mistake it for a new document and the previous version's
  chunks linger in the index, quietly still being retrieved.
* A **renamed** one produces a *different* ``doc_id`` for content that is
  already stored. Mistake it for a new document and the same text is indexed
  twice, so every retrieval returns it twice.

Detection, not policy
---------------------
:class:`ChangeKind.RENAMED` reports a *fact* — this content already exists
under a different path — and deliberately stops there. Content alone cannot
tell a rename from a move from a plain copy: ``spec.md`` -> ``api-spec.md``
and ``spec.md`` -> ``backup/spec.md`` produce identical hashes, yet one
should probably redirect to the existing document while the other should
probably be indexed independently. That call belongs to the upload API and
the ingest pipeline, which know about users and intent; this module reports
what the bytes say and lets them decide.

Pure functions over what is already known
-----------------------------------------
Nothing here opens a file, talks to a store, or holds state. Callers pass
the documents they know about and get a decision back, which is what makes
this testable with no database, no network and no API key. The lookups are
named ``find_by_doc_id`` / ``find_by_hash`` to mirror the registry Protocol
in ARCHITECTURE section 5.4, so when the real registry lands it can replace these
list scans with indexed queries without changing this contract. Until then
a scan is a linear pass — fine for what a single-process ingest sees, and
said here rather than left to be discovered.

**The one rule callers must not get wrong:** hash the *same* input that was
used to build the stored documents. :meth:`Document.create` hashes the
extracted text it is given unless you pass ``content_hash``; a digest only
ever matches what it was computed over, so comparing a file's bytes against
a text-derived hash would silently report every file as modified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

# Re-exported so callers have one import site for the identity helpers that
# deduplication is meaningless without -- the same reason loaders.py
# re-exports load_pdf.
from app.domain.models import Document, content_sha256, make_doc_id


class ChangeKind(StrEnum):
    """What an incoming file is, relative to what is already known.

    ``StrEnum`` so it serialises straight to JSON and compares equal to its
    string form (``decision.kind == "modified"`` works).
    """

    #: Neither this path nor this content has been seen before.
    NEW = "new"
    #: Same path and same content — already indexed, nothing to do.
    UNCHANGED = "unchanged"
    #: Same path, different content — edited since it was ingested.
    MODIFIED = "modified"
    #: Different path, same content — a rename, move or copy.
    RENAMED = "renamed"


@dataclass(frozen=True, slots=True)
class DedupDecision:
    """The verdict for one incoming file against the known documents.

    Immutable, like every model here, so a decision cannot be altered after
    a caller has acted on it.

    Parameters
    ----------
    kind:
        Which row of the table in the module docstring this is.
    doc_id:
        The id the incoming ``source_path`` hashes to. Always present —
        even for :attr:`ChangeKind.NEW`, because callers need the id they
        are about to create a document under.
    content_sha256:
        Hash of the incoming content. Present for the same reason.
    existing_doc_id:
        The **known** record this decision concerns: the document already
        stored at this path for ``unchanged``/``modified``, and the document
        holding the same content for ``renamed``. ``None`` for ``new``. For
        ``renamed`` this differs from :attr:`doc_id` — that difference is
        the whole point of detecting it.
    reason:
        One human-readable line saying what happened, safe to log. Contains
        ids but never file paths or document text.
    """

    kind: ChangeKind
    doc_id: str
    content_sha256: str
    existing_doc_id: str | None = None
    reason: str = ""

    @property
    def is_new(self) -> bool:
        """True when neither the path nor the content is known."""
        return self.kind is ChangeKind.NEW

    @property
    def is_unchanged(self) -> bool:
        """True when this exact document has already been indexed."""
        return self.kind is ChangeKind.UNCHANGED

    @property
    def is_modified(self) -> bool:
        """True when a known document's content has changed."""
        return self.kind is ChangeKind.MODIFIED

    @property
    def is_renamed(self) -> bool:
        """True when this content already lives under a different path."""
        return self.kind is ChangeKind.RENAMED

    @property
    def needs_ingest(self) -> bool:
        """True when work remains: a new document, or an edited one.

        Covers both because both end in writing chunks — a new document
        writes them for the first time, an edited one replaces what is
        already there.
        """
        return self.kind in {ChangeKind.NEW, ChangeKind.MODIFIED}

    @property
    def is_duplicate_content(self) -> bool:
        """True when this content is already present in the known set.

        What an upload endpoint checks to answer *"have I seen this file
        before?"* — either it is this very document unchanged, or the same
        bytes under another name.
        """
        return self.kind in {ChangeKind.UNCHANGED, ChangeKind.RENAMED}

    def summary(self) -> dict[str, str | None]:
        """JSON-safe summary for logs and API responses.

        Deliberately omits ``content_sha256``, matching
        :meth:`app.domain.models.Document.summary` — the project's stated
        position is that neither document text nor its digest belongs in a
        log line. Ids are included because they are how a log line points at
        the record involved.
        """
        return {
            "kind": str(self.kind),
            "doc_id": self.doc_id,
            "existing_doc_id": self.existing_doc_id,
            "reason": self.reason,
        }


def find_by_doc_id(known: Sequence[Document], doc_id: str) -> Document | None:
    """Return the known document with this id, or ``None``.

    Named to mirror ``DocumentRegistry.get``; a linear scan today, an
    indexed lookup tomorrow, same contract.
    """
    for document in known:
        if document.doc_id == doc_id:
            return document
    return None


def find_by_hash(known: Sequence[Document], digest: str) -> Document | None:
    """Return the first known document with this content hash, or ``None``.

    Named to mirror ``DocumentRegistry.find_by_hash``. "First" is the first
    in sequence order, so results are deterministic; if a corpus already
    contains two documents with identical content, either could be reported
    as the match, which is harmless because they are indistinguishable by
    content anyway.
    """
    for document in known:
        if document.content_sha256 == digest:
            return document
    return None


def classify(
    known: Sequence[Document],
    *,
    source_path: str,
    content: bytes | str,
) -> DedupDecision:
    """Decide whether an incoming file is new, unchanged, modified or renamed.

    Parameters
    ----------
    known:
        Documents already held — from the registry once it exists, or any
        list of :class:`~app.domain.models.Document` today. An empty
        sequence simply means everything is new.
    source_path:
        Path of the incoming file, in the same terms the stored documents
        use (relative to the data directory, forward slashes). It is what
        ``doc_id`` is derived from.
    content:
        The incoming content — file bytes or extracted text, accepted the
        same way :func:`~app.domain.models.content_sha256` is. **Pass the
        same kind of value that built the stored documents**, or the hash
        will never match and every file will read as modified.

    Returns
    -------
    DedupDecision
        Always a decision; the four kinds are outcomes, not errors.

    Raises
    ------
    ValidationError
        If ``source_path`` is empty, absolute, or contains ``..`` — the
        same checks :func:`~app.domain.models.make_doc_id` always applies.
        This propagates deliberately: a malformed path is a caller bug, and
        reporting it as "this file is new" would quietly mint an id from a
        path that should never have reached here.

    Notes
    -----
    The path is checked **before** the content. When a document already
    exists at this path, that is the decision that matters: re-ingesting
    under its existing ``doc_id`` is correct whether or not the new bytes
    happen to equal some other document's, so the content lookup is skipped
    rather than allowed to reinterpret an edit as a rename.
    """
    doc_id = make_doc_id(source_path)
    digest = content_sha256(content)

    existing_at_path = find_by_doc_id(known, doc_id)
    if existing_at_path is not None:
        if existing_at_path.content_sha256 == digest:
            return DedupDecision(
                kind=ChangeKind.UNCHANGED,
                doc_id=doc_id,
                content_sha256=digest,
                existing_doc_id=existing_at_path.doc_id,
                reason=(f"document {doc_id} already holds this exact content; nothing to ingest"),
            )
        return DedupDecision(
            kind=ChangeKind.MODIFIED,
            doc_id=doc_id,
            content_sha256=digest,
            existing_doc_id=existing_at_path.doc_id,
            reason=(
                f"document {doc_id} exists but its content changed; "
                "re-ingesting replaces its previous chunks"
            ),
        )

    elsewhere = find_by_hash(known, digest)
    if elsewhere is not None:
        return DedupDecision(
            kind=ChangeKind.RENAMED,
            doc_id=doc_id,
            content_sha256=digest,
            existing_doc_id=elsewhere.doc_id,
            reason=(
                f"this content is already indexed as {elsewhere.doc_id}; "
                "same bytes reached a different path"
            ),
        )

    return DedupDecision(
        kind=ChangeKind.NEW,
        doc_id=doc_id,
        content_sha256=digest,
        existing_doc_id=None,
        reason=f"neither {doc_id} nor this content is known yet",
    )


__all__ = [
    "ChangeKind",
    "DedupDecision",
    "content_sha256",
    "find_by_doc_id",
    "find_by_hash",
    "make_doc_id",
]
