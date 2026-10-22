"""PDF builders, re-exported from :mod:`scripts.pdf_fixtures`.

The builders themselves live in ``scripts/`` because two consumers need them
and neither may import the other: tests build PDFs to assert against, and the
sample-data generator writes them to disk. One copy in the tooling package
means tests do not depend on tooling code — or vice versa.

This module stays as the import site so existing tests keep their line:
``from tests.unit.pdf_fixtures import build_pdf``.
"""

from __future__ import annotations

from scripts.pdf_fixtures import (
    build_blank_pdf,
    build_encrypted_pdf,
    build_pdf,
    build_pdf_with_broken_page_tree,
    build_pdf_with_metadata,
)

__all__ = [
    "build_blank_pdf",
    "build_encrypted_pdf",
    "build_pdf",
    "build_pdf_with_broken_page_tree",
    "build_pdf_with_metadata",
]
