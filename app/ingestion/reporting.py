"""Turn a batch of ingest results into a report someone can act on.

The ingest service (:mod:`app.ingestion.service`) already answers *"what is
the state of this batch?"* — and deliberately keeps that answer small,
because its ``summary()`` is a payload for **logs** and **API responses**,
where a long per-file breakdown is noise with a cost attached.

A report is a different artefact: something produced once at the end of a
run, read by a person deciding what to fix, or written to a file and
committed alongside sample data. It can afford detail, and it should group
problems rather than list them one by one, because the useful question after
a 200-file ingest is never *"which files failed?"* — it is *"and how many
were the same failure?"* A hundred ``encrypted_pdf`` failures is one problem
with a hundred instances, and the tallies here say so at a glance.

So this module **derives** from ``IngestResult``; it never re-reads a file
and never becomes a second source of truth. Rebuilding it after every change
to a batch costs nothing and removes the question of which copy is current.

The same reasoning produces a second, smaller artefact: the **index
report**, derived from section 5.17's ``IndexResult``. Its counts answer a write
pass's question — created, replaced, chunks written — while its groupings
come from exactly the same ``CodeTally`` buckets, so a reader who learned
the ingest report reads this one without a legend.

What it adds beyond the counts
------------------------------

* Problem **grouping** — failures and skips bucketed by their stable
  ``code``, ordered by frequency, each carrying the paths that produced it.
* **Bytes read** — an ingest's cost driver, which no count conveys.
* Optional **deduplication tallies** — how many files were new, unchanged,
  modified or renamed (section 5.2.6), the answer to *"did this run actually
  change anything?"* Supplied by the caller rather than computed here,
  because content hashing needs bytes this module never opens.
* A human-readable :meth:`IngestReport.render` for terminals and sample
  reports, alongside the machine-readable :meth:`IngestReport.to_dict`.

**No document text, ever.** Every value comes from counts, codes and
relative paths — the same rule ``FileResult.summary()`` and
``IngestResult.summary()`` already follow, and the reason all three are safe
to write to a log. ``root`` is the scanned directory's *name*, never an
absolute path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.ingestion.deduplication import ChangeKind, DedupDecision
from app.ingestion.indexer import IndexOutcome, IndexResult
from app.ingestion.service import FileResult, IngestResult

__all__ = [
    "UNSPECIFIED_CODE",
    "CodeTally",
    "IndexReport",
    "IngestReport",
    "build_index_report",
    "build_report",
    "format_bytes",
]

#: Fallback label for a non-loaded result that somehow carried no code.
#: The service always sets one; this keeps the report honest instead of
#: crashing on a result built elsewhere.
UNSPECIFIED_CODE = "unspecified"

_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


@dataclass(frozen=True, slots=True)
class CodeTally:
    """One failure or skip code, with how often it occurred and where.

    Parameters
    ----------
    code:
        The stable machine-readable reason (``encrypted_pdf``), or
        :data:`UNSPECIFIED_CODE` if a result arrived without one.
    count:
        How many files produced this code — ``len(paths)``.
    paths:
        Relative paths in discovery order. Kept complete rather than capped:
        this is a report, and a truncated list of failures is a list you
        cannot act on.
    """

    code: str
    count: int
    paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form."""
        return {"code": self.code, "count": self.count, "paths": list(self.paths)}


@dataclass(frozen=True, slots=True)
class IngestReport:
    """A batch reduced to the figures a report needs.

    Immutable, like every other model here, so a report handed to a renderer
    cannot be rewritten underneath it.

    Parameters
    ----------
    root:
        Scanned directory's name only — never an absolute path.
    total_files / processed_files / indexable:
        Discovery's total, the files actually attempted, and the ones with
        text ready to index (which equals ``loaded``).
    loaded / skipped / failed / unsupported:
        The four outcome counts.
    total_bytes:
        Sum of the sizes learned, or ``None`` when no file reported one —
        never ``0``, because "we never measured" and "there was nothing" are
        different facts.
    failures / skips:
        :class:`CodeTally` groups, most frequent first then by code so the
        order is deterministic.
    dedup:
        ``(kind, count)`` pairs in ``ChangeKind`` order, or ``None`` when the
        caller supplied no decisions (deduplication not run).
    duration_seconds:
        Caller-measured wall time, or ``None`` when not supplied. Never
        measured here — a report must not invent a number.
    """

    root: str
    total_files: int
    processed_files: int
    indexable: int
    loaded: int
    skipped: int
    failed: int
    unsupported: int
    total_bytes: int | None
    failures: tuple[CodeTally, ...]
    skips: tuple[CodeTally, ...]
    dedup: tuple[tuple[str, int], ...] | None = None
    duration_seconds: float | None = None

    @property
    def is_empty(self) -> bool:
        """True when the scanned root held nothing recognisable."""
        return self.total_files == 0

    @property
    def has_failures(self) -> bool:
        """True when at least one file could not be read."""
        return self.failed > 0

    @property
    def total_problems(self) -> int:
        """Failed plus skipped files — everything that will not be indexed."""
        return self.failed + self.skipped

    def to_dict(self) -> dict[str, object]:
        """JSON-safe payload: counts, groupings and provenance only.

        Contains no document text, matching the rule the two ``summary()``
        methods already follow.
        """
        payload: dict[str, object] = {
            "root": self.root,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "indexable": self.indexable,
            "loaded": self.loaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "total_bytes": self.total_bytes,
            "failures": [tally.to_dict() for tally in self.failures],
            "skips": [tally.to_dict() for tally in self.skips],
            "duration_seconds": self.duration_seconds,
        }
        if self.dedup is not None:
            payload["dedup"] = dict(self.dedup)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise :meth:`to_dict` for writing to a report file."""
        return json.dumps(self.to_dict(), indent=indent)

    def render(self) -> str:
        """Human-readable plain text, for a terminal or a sample report.

        Deterministic and unstyled — no colour, no cursor control — so it
        reads the same in a CI log, a Windows console and a committed file.

        Returns the report without a trailing newline; ``print()`` supplies
        the one a terminal wants.
        """
        header = [f"Ingest report: {self.root}"]
        if self.is_empty:
            header.append("  no files found")
            return "\n".join(header)

        header.append(
            f"  {self.total_files} discovered"
            f" | {self.processed_files} attempted"
            f" | {self.indexable} ready to index"
        )
        header.append(_detail_line(self.total_bytes, self.duration_seconds))

        body = _count_rows(
            (
                ("loaded", self.loaded),
                ("skipped", self.skipped),
                ("failed", self.failed),
                ("unsupported", self.unsupported),
            )
        )
        body += _problem_sections(self.failures, self.skips)
        body += _dedup_section(self.dedup)

        return "\n".join(header + body)


@dataclass(frozen=True, slots=True)
class IndexReport:
    """The indexing pass reduced to the figures a report needs.

    The pass of section 5.17 *writes*, so its report counts writes: how many
    documents were created, how many replaced, and how many chunks
    actually reached the store — figures an ingest report cannot have,
    because loading a file says nothing about the index.

    Parameters
    ----------
    root:
        Scanned directory's name only — never an absolute path.
    total_files / processed_files:
        Discovery's total (supported and unsupported together) and the
        supported files the pass attempted.
    created / replaced / skipped / failed / unsupported:
        The outcome counts. ``skipped`` is the honest no-op bucket —
        unchanged, renamed, empty — rather than a verdict on the file.
    chunks_written:
        Chunks that reached the store during this pass.
    total_bytes:
        Sum of the sizes learned, or ``None`` when no file reported one —
        never ``0``, because "we never measured" and "there was nothing"
        are different facts.
    failures / skips:
        :class:`CodeTally` groups over the *pass's* codes: a failure that
        happened after a successful load (the provider, the registry)
        carries the pipeline's code even though the file read fine.
    dedup:
        ``(kind, count)`` pairs in ``ChangeKind`` order, or ``None`` when
        nothing in the batch reached classification.
    duration_seconds:
        Caller-measured wall time, or ``None`` when not supplied. Never
        measured here — a report must not invent a number.
    """

    root: str
    total_files: int
    processed_files: int
    created: int
    replaced: int
    skipped: int
    failed: int
    unsupported: int
    chunks_written: int
    total_bytes: int | None
    failures: tuple[CodeTally, ...]
    skips: tuple[CodeTally, ...]
    dedup: tuple[tuple[str, int], ...] | None = None
    duration_seconds: float | None = None

    @property
    def is_empty(self) -> bool:
        """True when the scanned root held nothing the pass recognised."""
        return self.total_files == 0

    @property
    def has_failures(self) -> bool:
        """True when at least one file did not make it into the index."""
        return self.failed > 0

    def to_dict(self) -> dict[str, object]:
        """JSON-safe payload: counts, groupings and provenance only."""
        payload: dict[str, object] = {
            "root": self.root,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "created": self.created,
            "replaced": self.replaced,
            "skipped": self.skipped,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "chunks_written": self.chunks_written,
            "total_bytes": self.total_bytes,
            "failures": [tally.to_dict() for tally in self.failures],
            "skips": [tally.to_dict() for tally in self.skips],
            "duration_seconds": self.duration_seconds,
        }
        if self.dedup is not None:
            payload["dedup"] = dict(self.dedup)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise :meth:`to_dict` for writing to a report file."""
        return json.dumps(self.to_dict(), indent=indent)

    def render(self) -> str:
        """Human-readable plain text, for a terminal or a sample report.

        Same shape and rules as :meth:`IngestReport.render`: deterministic,
        unstyled, and without a trailing newline.
        """
        header = [f"Index report: {self.root}"]
        if self.is_empty:
            header.append("  no files found")
            return "\n".join(header)

        header.append(
            f"  {self.total_files} discovered"
            f" | {self.processed_files} attempted"
            f" | {self.chunks_written} chunks written"
        )
        header.append(_detail_line(self.total_bytes, self.duration_seconds))

        body = _count_rows(
            (
                ("created", self.created),
                ("replaced", self.replaced),
                ("skipped", self.skipped),
                ("failed", self.failed),
                ("unsupported", self.unsupported),
            )
        )
        body += _problem_sections(self.failures, self.skips)
        body += _dedup_section(self.dedup)

        return "\n".join(header + body)


def build_report(
    result: IngestResult,
    *,
    decisions: Sequence[DedupDecision] | None = None,
    duration_seconds: float | None = None,
) -> IngestReport:
    """Summarise one batch as an :class:`IngestReport`.

    Parameters
    ----------
    result:
        The batch to summarise. Nothing is re-read: every figure comes from
        results already computed.
    decisions:
        Deduplication decisions for *this* batch, if deduplication was run.
        ``None`` (the default) leaves ``IngestReport.dedup`` as ``None``,
        which renders as no deduplication section rather than as zeros.
        Counts only — decisions need not be positionally aligned to files.
    duration_seconds:
        Wall time measured by the caller. Negative values raise, since they
        can only come from a broken clock and printing them would launder
        the error into the report.

    Returns
    -------
    IngestReport
    """
    if duration_seconds is not None and duration_seconds < 0:
        raise ValueError(f"duration_seconds must be >= 0, got {duration_seconds}")

    sizes = [f.byte_size for f in result.files if f.byte_size is not None]

    return IngestReport(
        root=result.root,
        total_files=result.total_count,
        processed_files=result.processed_count,
        indexable=result.indexable_count,
        loaded=result.loaded_count,
        skipped=result.skipped_count,
        failed=result.failed_count,
        unsupported=result.unsupported_count,
        total_bytes=sum(sizes) if sizes else None,
        failures=_tally(result.failed()),
        skips=_tally(result.skipped()),
        dedup=_dedup_tally(decisions) if decisions is not None else None,
        duration_seconds=duration_seconds,
    )


def build_index_report(
    result: IndexResult,
    *,
    duration_seconds: float | None = None,
) -> IndexReport:
    """Summarise one indexing pass as an :class:`IndexReport`.

    Parameters
    ----------
    result:
        The pass to summarise. Nothing is re-read: every figure comes from
        outcomes already computed.
    duration_seconds:
        Wall time measured by the caller. Negative values raise, since they
        can only come from a broken clock and printing them would launder
        the error into the report.

    Returns
    -------
    IndexReport
    """
    if duration_seconds is not None and duration_seconds < 0:
        raise ValueError(f"duration_seconds must be >= 0, got {duration_seconds}")

    sizes = [outcome.file.byte_size for outcome in result.outcomes]
    known_sizes = [size for size in sizes if size is not None]

    # Failures and skips are tallied by the *outcome's* code, not the
    # load's: a file that read perfectly and then hit a provider outage
    # must group under the outage's code, which only the outcome carries.
    return IndexReport(
        root=result.root,
        total_files=result.total_count,
        processed_files=len(result.outcomes),
        created=result.created_count,
        replaced=result.replaced_count,
        skipped=result.skipped_count,
        failed=result.failed_count,
        unsupported=len(result.unsupported),
        chunks_written=result.chunks_written,
        total_bytes=sum(known_sizes) if known_sizes else None,
        failures=_outcome_tally([o for o in result.outcomes if o.is_failed]),
        skips=_outcome_tally([o for o in result.outcomes if o.is_skipped]),
        dedup=_dedup_tally(result.decisions) if result.decisions else None,
        duration_seconds=duration_seconds,
    )


def format_bytes(size: int | None) -> str:
    """Render a byte count with binary units: ``1536`` -> ``"1.5 KiB"``.

    ``None`` becomes ``"unknown"`` rather than ``0 B`` for the reason given
    on :attr:`IngestReport.total_bytes`.

    Uses KiB/MiB/GiB rather than KB/MB/GB because those labels are
    unambiguous — 1 KiB is 1024 bytes by definition, whereas "MB" is read
    both ways in the wild and this project would rather be precise than
    familiar.

    Raises
    ------
    ValueError
        For a negative size, which no file can have.
    """
    if size is None:
        return "unknown"
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")

    value = float(size)
    for unit in _UNITS:
        if value < 1024 or unit == _UNITS[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def _tally(files: Iterable[FileResult]) -> tuple[CodeTally, ...]:
    """Bucket non-loaded results by code, most frequent first.

    Ties break on the code itself so two runs over the same data produce
    byte-identical reports.
    """
    return _bucket((file_result.code, file_result.relative_path) for file_result in files)


def _outcome_tally(outcomes: Iterable[IndexOutcome]) -> tuple[CodeTally, ...]:
    """Bucket pass outcomes by *their* code, most frequent first.

    Same ordering rules as :func:`_tally`, but reading the outcome rather
    than the load result: the pass records failures that happened long
    after a file loaded successfully (the provider, the registry), and
    those codes exist nowhere else.
    """
    return _bucket((outcome.code, outcome.file.relative_path) for outcome in outcomes)


def _bucket(code_and_path: Iterable[tuple[str | None, str]]) -> tuple[CodeTally, ...]:
    """Group ``(code, path)`` pairs, most frequent first, ties by code."""
    buckets: dict[str, list[str]] = {}
    for code, path in code_and_path:
        buckets.setdefault(code or UNSPECIFIED_CODE, []).append(path)

    tallies = [
        CodeTally(code=code, count=len(paths), paths=tuple(paths))
        for code, paths in buckets.items()
    ]
    tallies.sort(key=lambda tally: (-tally.count, tally.code))
    return tuple(tallies)


def _detail_line(total_bytes: int | None, duration_seconds: float | None) -> str:
    """The "size read | duration" line both reports share."""
    parts: list[str] = []
    if total_bytes is not None:
        parts.append(f"{format_bytes(total_bytes)} read")
    else:
        parts.append("size unknown")
    if duration_seconds is not None:
        parts.append(f"{duration_seconds:.2f}s")
    return "  " + " | ".join(parts)


def _count_rows(rows: Sequence[tuple[str, int]]) -> list[str]:
    """Render ``(label, value)`` pairs with aligned labels."""
    width = max(len(label) for label, _ in rows)
    return [f"  {label:<{width}}  {value}" for label, value in rows]


def _problem_sections(
    failures: tuple[CodeTally, ...],
    skips: tuple[CodeTally, ...],
) -> list[str]:
    """The ``failed by code`` / ``skipped by code`` blocks, empty when there is nothing to say."""
    body: list[str] = []
    for title, tallies in (("failed by code", failures), ("skipped by code", skips)):
        if not tallies:
            continue
        body.append("")
        body.append(f"  {title}")
        for tally in tallies:
            body.append(f"    {tally.code}  x{tally.count}")
            body.extend(f"      {path}" for path in tally.paths)
    return body


def _dedup_section(dedup: tuple[tuple[str, int], ...] | None) -> list[str]:
    """The ``deduplication`` block, empty when the caller reported none."""
    if dedup is None:
        return []
    return [
        "",
        "  deduplication",
        "    " + ", ".join(f"{kind} {count}" for kind, count in dedup),
    ]


def _dedup_tally(
    decisions: Sequence[DedupDecision],
) -> tuple[tuple[str, int], ...]:
    """Count decisions per :class:`ChangeKind`, in enum declaration order.

    Zero counts are kept: a report that omits ``renamed`` and one that
    reports ``renamed 0`` say different things to a reader scanning the
    output.
    """
    counts = {str(kind): 0 for kind in ChangeKind}
    for decision in decisions:
        counts[str(decision.kind)] += 1
    return tuple(counts.items())
