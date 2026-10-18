"""PDF builders for tests (Stage 12).

The PDFs are generated in memory rather than checked into the repository.
Binary fixtures are opaque in review, they bloat the history, and a corrupted
one is almost impossible to spot — building them from a few dozen lines of
explicit PDF syntax keeps the tests readable and self-contained.

The syntax is the minimum that pypdf will parse: a catalog, a page tree, one
content stream per page, one font, and a correct cross-reference table.
"""

from __future__ import annotations

import io


def _assemble(objects: list[bytes]) -> bytes:
    """Write PDF objects with a correct xref table and trailer."""
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode())
        out.write(obj)
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(f"startxref\n{xref_pos}\n%%EOF\n".encode())
    return out.getvalue()


def build_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one Helvetica text run per page.

    Object layout: 1 = catalog, 2 = page tree, then two objects per page
    (the page and its content stream), with the font last.
    """
    objects: list[bytes] = []
    n_pages = len(pages)
    font_obj_num = 3 + 2 * n_pages

    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())

    for i, text in enumerate(pages):
        page_num = 3 + 2 * i
        content_num = page_num + 1
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET\n"
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_num} 0 R "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>"
            ).encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream".encode())

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    return _assemble(objects)


def build_blank_pdf(n_pages: int = 1) -> bytes:
    """A PDF with real pages but no content streams.

    Models a scanned document: the page objects exist, so the file is
    structurally valid, but there is no text layer to extract.
    """
    objects: list[bytes] = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for _ in range(n_pages):
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
    return _assemble(objects)


def build_pdf_with_broken_page_tree() -> bytes:
    """A PDF whose catalog points at a ``/Pages`` object that does not exist.

    The file parses, but resolving the page tree fails. That is the case pypdf
    reports when the page count is requested rather than at open time, which
    is exactly the path the loader's page-count error handling covers.
    """
    return _assemble([b"<< /Type /Catalog /Pages 9 0 R >>"])


def build_encrypted_pdf(pages: list[str], password: str) -> bytes:
    """A valid PDF encrypted with a user password."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.append(io.BytesIO(build_pdf(pages)))
    writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def build_pdf_with_metadata(pages: list[str], metadata: dict[str, str]) -> bytes:
    """A valid PDF carrying a document-info dictionary.

    Values are passed straight through to pypdf's writer, which is the point:
    the tests need to see what a *real* producer's metadata survives, not what
    a hand-written dictionary claims it looks like. That includes control
    characters, over-long values and malformed dates.
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.append(io.BytesIO(build_pdf(pages)))
    if metadata:
        writer.add_metadata(metadata)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
