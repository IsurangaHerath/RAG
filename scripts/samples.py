"""Prepare a small, deterministic sample corpus for the ingest pipeline.

Why generate rather than commit
-------------------------------
``data/`` is git-ignored (binary state has no place in review), and a
corpus that only exists on one machine cannot be discussed in an issue or
recreated by a newcomer. So the sample documents are *written on demand*
from source kept here, where every byte is reviewable and the whole corpus
costs a handful of lines.

**Determinism is a requirement, not a nicety.** :mod:`app.ingestion`
identifies a document by hashing its path and hashing its content (section 5.1), so
regenerating a sample that differs by a single byte would make every
re-run report the corpus as *modified* and re-index all of it. Everything
here is therefore a fixed literal — no timestamps, no random ids — and each
builder is byte-stable across runs, including the encrypted one.

Two areas, one for each half of what the ingest service reports
--------------------------------------------------------------
``corpus/`` is the useful half: readable documents a later retrieval stage
can be evaluated against. ``edge-cases/`` exists to *fail honestly* — a
blank file, a scanned PDF, a locked PDF, a non-UTF-8 export and an
unsupported image, which between them exercise ``empty_file``,
``no_text_layer``, ``encrypted_pdf``, ``binary_content`` and the
unsupported list without anyone having to break real data by hand.

Never deletes: a file that already exists but differs is left alone unless
``force`` is set, so running the command can never destroy something a user
edited in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.pdf_fixtures import build_blank_pdf, build_encrypted_pdf, build_pdf

__all__ = [
    "PlannedFile",
    "PrepareResult",
    "prepare",
    "sample_files",
]

# Shared between the FAQ and the API reference on purpose: two documents
# stating the same fact is what makes a later retrieval stage worth testing.
_RATE_LIMIT_TEXT = (
    "Each account may authorise 100 payments per second. Beyond that the "
    "API answers HTTP 429 with a Retry-After header giving the seconds to "
    "wait. Retrying the same payment identifier is safe, because "
    "authorisation is idempotent."
)

_API_SPEC = """\
# PayLink API Reference

PayLink authorises payments between a merchant and a payer. Every request
carries a bearer token issued from the dashboard, and every response
carries a request id you can quote to support.

## Base URL

    https://api.paylink.example/v1

## Endpoints

### POST /payments/authorize

Authorises an amount against a card or bank account.

| Field | Type | Notes |
| --- | --- | --- |
| `amount` | integer | Minor units: 1000 is 10.00 USD |
| `currency` | string | Three-letter ISO code, uppercase |
| `reference` | string | Your own identifier, up to 64 characters |

### GET /payments/{id}

Returns one payment with its current state: pending, captured, refunded
or failed.

## Rate limits

__RATE_LIMIT__

## Errors

| Code | Status | Meaning |
| --- | --- | --- |
| `invalid_amount` | 400 | Amount missing or negative |
| `insufficient_funds` | 402 | Payer could not cover the amount |
| `rate_limited` | 429 | Too many requests this second |
""".replace("__RATE_LIMIT__", _RATE_LIMIT_TEXT)

_DEPLOYMENT_GUIDE = """\
# Deploying PayLink

PayLink ships as a single container with no system dependencies beyond a
certificate store.

## Requirements

* 2 CPU cores and 2 GB of memory for 50 authorisations per second
* A Postgres 15 instance for ledger storage
* TLS terminated at the load balancer

## Steps

1. Set the environment variables listed below.
2. Run database migrations with `paylink migrate`.
3. Start the API with `paylink serve --port 8080`.
4. Confirm health at `/healthz`; it reports ok once the ledger connects.

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | yes | Ledger connection string |
| `PAYLINK_TOKEN_SECRET` | yes | Signs bearer tokens |
| `LOG_LEVEL` | no | Defaults to info |

## Scaling

Authorisation is stateless, so scale the API horizontally behind the load
balancer. Ledger writes serialise through Postgres; past 200 authorisations
per second move the ledger onto its own instance.
"""

_FAQ = """\
PayLink Frequently Asked Questions

Q: How long does an authorisation take?
A: The API answers within 200 milliseconds at the 95th percentile.
   Capturing an authorised payment later is a separate call.

Q: What happens when I exceed the rate limit?
A: __RATE_LIMIT__

Q: Which currencies are supported?
A: Any three-letter ISO 4217 code your account has enabled. Amounts are
   always sent in minor units, so 1000 means 10.00 USD.

Q: Can I refund a captured payment?
A: Yes, with POST /payments/{id}/refund, up to 90 days after capture.

Q: Where do I find the request id for a support ticket?
A: Every response carries it as the X-Request-Id header and in the body.
""".replace("__RATE_LIMIT__", _RATE_LIMIT_TEXT)

_RELEASE_NOTES = """\
# Release Notes

## 2.4.0 - 2026-03-10
* Refunds can now be partial; the remainder stays captured.
* GET /payments/{id} gained a refunded_amount field.

## 2.3.1 - 2026-01-22
* Fixed a bug where a retried authorisation could be recorded twice when
  the connection dropped after the ledger write.

## 2.3.0 - 2025-12-02
* Rate limits moved from 50 to 100 authorisations per second.
* Retry-After is now sent with every 429 response.

## 2.2.0 - 2025-09-15
* Webhooks for payment.captured and payment.failed.
* Dashboard tokens can be scoped to a single account.
"""

# Text passed to build_pdf lands inside a PDF literal string, so it must
# avoid parentheses and backslashes (unescaped by that builder) and stay
# ASCII (Helvetica is decoded as Latin-1 on extraction).
_ARCHITECTURE_PAGES = [
    "PayLink Architecture Overview. The API layer is stateless. It reads "
    "configuration at start up and forwards every authorisation to the "
    "ledger service. The ledger writes to Postgres and publishes an event "
    "for each captured payment.",
    "Storage. Postgres holds the ledger of payments and their states. "
    "Rows are append only: a refund inserts a new row rather than editing "
    "the original capture, so the history stays complete and can be "
    "replayed.",
    "Reliability. The API retries a ledger write three times with a "
    "backoff. If the write still fails the payment is marked failed and an "
    "event is published, so clients see a stable error code rather than a "
    "partial result.",
]

_LOCKED_RUNBOOK_PAGES = [
    "Operations runbook. Restart the ledger before the API, never after.",
    "Rotate signing keys every ninety days.",
]

# A structurally valid PNG signature plus an IHDR. Discovery rejects it on
# its extension alone, so the bytes are never read -- written only so the
# file is not a misleading empty placeholder.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
)


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """One file the sample corpus will write.

    Parameters
    ----------
    relative_path:
        Forward-slashed path below the destination directory. The
        sub-directory is what separates the readable corpus from the files
        that are meant to fail.
    data:
        Exact bytes to write. Fixed literals, so a second run writes the
        identical file and ingestion reports it unchanged.
    note:
        What this sample demonstrates, shown when the file is written.
    """

    relative_path: str
    data: bytes
    note: str

    @property
    def path(self) -> Path:
        """Path as an OS object — forward slashes are not portable on Windows."""
        return Path(*self.relative_path.split("/"))


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """What a run over the sample corpus did, file by file.

    Parameters
    ----------
    destination:
        Directory the files were written under.
    created:
        Written for the first time, in corpus order.
    unchanged:
        Already present with exactly the planned bytes.
    updated:
        Replaced because they differed and ``force`` was set.
    kept:
        Differed but were left alone. The only state that needs a decision.
    """

    destination: Path
    created: tuple[str, ...]
    unchanged: tuple[str, ...]
    updated: tuple[str, ...]
    kept: tuple[str, ...]

    @property
    def total(self) -> int:
        """How many files the corpus contains."""
        return len(self.created) + len(self.unchanged) + len(self.updated) + len(self.kept)

    def summary(self) -> dict[str, object]:
        """JSON-safe summary of the run."""
        return {
            "destination": str(self.destination),
            "total": self.total,
            "created": list(self.created),
            "unchanged": list(self.unchanged),
            "updated": list(self.updated),
            "kept": list(self.kept),
        }

    def render(self) -> str:
        """Plain text, one line per file, with the outcome marked."""
        lines = [f"Sample data in {self.destination}", f"  {self.total} files"]
        for label, paths, marker in (
            ("created", self.created, "+"),
            ("unchanged", self.unchanged, "="),
            ("updated", self.updated, "~"),
            ("kept", self.kept, "!"),
        ):
            for path in paths:
                suffix = " (differs; use --force to replace)" if label == "kept" else ""
                lines.append(f"  {marker} {label:<10} {path}{suffix}")
        if self.kept:
            lines.append(f"  {len(self.kept)} file(s) already differ and were left alone.")
        return "\n".join(lines)


def sample_files() -> tuple[PlannedFile, ...]:
    """Every file the corpus writes, in a stable order.

    Called by :func:`prepare` and by tests, so the plan and the assertions
    about it can never drift apart.
    """
    return (
        PlannedFile(
            "corpus/api-spec.md",
            _API_SPEC.encode("utf-8"),
            "endpoints, errors and rate limits",
        ),
        PlannedFile(
            "corpus/deployment-guide.md",
            _DEPLOYMENT_GUIDE.encode("utf-8"),
            "requirements, scaling advice",
        ),
        PlannedFile(
            "corpus/faq.txt",
            _FAQ.encode("utf-8"),
            "plain text, and a fact repeated from the API reference",
        ),
        PlannedFile(
            "corpus/release-notes.md",
            _RELEASE_NOTES.encode("utf-8"),
            "dated history for a retrieval test",
        ),
        PlannedFile(
            "corpus/architecture.pdf",
            build_pdf(_ARCHITECTURE_PAGES),
            "three-page PDF with extractable text",
        ),
        PlannedFile(
            "edge-cases/empty-notes.txt",
            b"",
            "skipped as empty_file",
        ),
        PlannedFile(
            "edge-cases/scanned-contract.pdf",
            build_blank_pdf(2),
            "two-page scan with no text layer -> no_text_layer",
        ),
        PlannedFile(
            "edge-cases/locked-runbook.pdf",
            build_encrypted_pdf(_LOCKED_RUNBOOK_PAGES, "sample-password"),
            "password-protected -> pdf_encrypted failure",
        ),
        PlannedFile(
            "edge-cases/broken-export.txt",
            b"id,name,amount\r\n\x00\x00\x00export truncated\r\n",
            "contains NUL bytes -> binary_content failure",
        ),
        PlannedFile(
            "edge-cases/logo.png",
            _PNG_BYTES,
            "extension the loaders do not handle -> unsupported",
        ),
    )


def prepare(destination: Path, *, force: bool = False) -> PrepareResult:
    """Write the sample corpus under ``destination``.

    Each planned file is compared against what is already there, so a
    repeat run over an untouched corpus reports every file unchanged and
    writes nothing.

    Parameters
    ----------
    destination:
        Directory to write under. Created, along with any missing parents.
    force:
        Replace files that exist with different bytes. Without it they are
        reported as *kept* and left exactly as they were.

    Returns
    -------
    PrepareResult
        Never raises for expected conditions: a file that cannot be
        replaced simply stays. Genuine I/O faults still propagate, because
        a bug must not be reported as a tidy summary.

    Notes
    -----
    Files not listed in :func:`sample_files` are never removed. This writes,
    it does not manage a directory.
    """
    destination.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    unchanged: list[str] = []
    updated: list[str] = []
    kept: list[str] = []

    for planned in sample_files():
        target = destination / planned.path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(planned.data)
            created.append(planned.relative_path)
        elif target.is_file() and target.read_bytes() == planned.data:
            unchanged.append(planned.relative_path)
        elif force:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(planned.data)
            updated.append(planned.relative_path)
        else:
            kept.append(planned.relative_path)

    return PrepareResult(
        destination=destination,
        created=tuple(created),
        unchanged=tuple(unchanged),
        updated=tuple(updated),
        kept=tuple(kept),
    )
