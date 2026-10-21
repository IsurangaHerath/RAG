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

What it adds beyond the counts
------------------------------

* Problem **grouping** — failures and skips bucketed by their stable
  ``code``, ordered by frequency, each carrying the paths that produced it.
* **Bytes read** — an ingest's cost driver, which no count conveys.
* Optional **deduplication tallies** — how many files were new, unchanged,
  modified or renamed (§5.2.6), the answer to *"did this run actually
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
from app.ingestion.service import FileResult, IngestResult

__all__ = [
    "UNSPECIFIED_CODE",
    "CodeTally",
    "IngestReport",
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
        detail: list[str] = []
        if self.total_bytes is not None:
            detail.append(f"{format_bytes(self.total_bytes)} read")
        else:
            detail.append("size unknown")
        if self.duration_seconds is not None:
            detail.append(f"{self.duration_seconds:.2f}s")
        header.append("  " + " | ".join(detail))

        rows = (
            ("loaded", self.loaded),
            ("skipped", self.skipped),
            ("failed", self.failed),
            ("unsupported", self.unsupported),
        )
        width = max(len(label) for label, _ in rows)
        body = [f"  {label:<{width}}  {value}" for label, value in rows]

        problem_sections = (
            ("failed by code", self.failures),
            ("skipped by code", self.skips),
        )
        for title, tallies in problem_sections:
            if not tallies:
                continue
            body.append("")
            body.append(f"  {title}")
            for tally in tallies:
                body.append(f"    {tally.code}  x{tally.count}")
                body.extend(f"      {path}" for path in tally.paths)

        if self.dedup is not None:
            body.append("")
            body.append("  deduplication")
            body.append("    " + ", ".join(f"{kind} {count}" for kind, count in self.dedup))

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
    buckets: dict[str, list[str]] = {}
    for file_result in files:
        buckets.setdefault(file_result.code or UNSPECIFIED_CODE, []).append(
            file_result.relative_path
        )

    tallies = [
        CodeTally(code=code, count=len(paths), paths=tuple(paths))
        for code, paths in buckets.items()
    ]
    tallies.sort(key=lambda tally: (-tally.count, tally.code))
    return tuple(tallies)


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
