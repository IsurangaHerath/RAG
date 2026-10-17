"""File discovery for ingestion (Stage 10).

Finds candidate documents under a root directory and classifies them as
supported or unsupported, without ever reading file *contents* — that is the
loaders' job (Stages 11-12). Discovery only looks at names, extensions and
sizes, so it is cheap and can be run repeatedly.

Two safety properties this module is responsible for:

**Containment** — every returned path resolves to somewhere inside the root.
A file reached through a symlink that points outside (or a path containing
``..``) is never returned. Without this, a crafted directory could make the
pipeline read ``/etc/shadow`` and index it.

**No surprises** — hidden entries (``.gitkeep``, ``.DS_Store``, ``.git``) are
skipped silently rather than reported as "unsupported documents", because
they are artefacts of the repository, not documents a user forgot to convert.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.domain.errors import NotFoundError, ValidationError
from app.domain.models import EXTENSION_TO_MEDIA_TYPE

#: Directories never worth descending into, whatever the user put in them.
#: Skipping these avoids walking ``.git`` or ``__pycache__`` on every run.
IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}
)


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """A supported document found on disk, ready to be handed to a loader."""

    #: Absolute path. Guaranteed to resolve inside the discovery root.
    path: Path
    #: Path relative to the root, POSIX-normalised and safe to store as an
    #: identifier (this is what :func:`app.domain.models.make_doc_id` hashes).
    relative_path: str
    #: Basename, e.g. ``api-spec.md``.
    filename: str
    #: MIME type from :data:`EXTENSION_TO_MEDIA_TYPE`.
    media_type: str
    #: Size in bytes, read from the directory entry (no file is opened).
    size_bytes: int

    @property
    def suffix(self) -> str:
        """Lower-cased extension including the dot, e.g. ``.md``."""
        return PurePosixPath(self.relative_path).suffix.lower()


@dataclass(frozen=True, slots=True)
class UnsupportedFile:
    """A file that is not a document this pipeline can load."""

    relative_path: str
    filename: str
    #: Lower-cased extension including the dot; ``""`` when the file has none.
    suffix: str
    #: Human-readable explanation, safe to show a user.
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of one discovery pass.

    An empty directory is a *result*, not an error: callers decide whether
    "nothing to do" is worth failing over.
    """

    #: Root that was scanned, resolved and absolute.
    root: Path
    files: tuple[DiscoveredFile, ...]
    unsupported: tuple[UnsupportedFile, ...]

    @property
    def is_empty(self) -> bool:
        """True when the root contains no document-like files at all.

        Hidden artefacts do not count, so a directory holding only
        ``.gitkeep`` is correctly reported as empty.
        """
        return not self.files and not self.unsupported

    @property
    def supported_count(self) -> int:
        """Number of loadable documents found."""
        return len(self.files)

    @property
    def unsupported_count(self) -> int:
        """Number of non-loadable files found."""
        return len(self.unsupported)

    @property
    def total_size_bytes(self) -> int:
        """Combined size of all supported documents."""
        return sum(f.size_bytes for f in self.files)

    def relative_paths(self) -> tuple[str, ...]:
        """Supported relative paths in scan order — handy for stable tests."""
        return tuple(f.relative_path for f in self.files)


def _is_hidden(name: str) -> bool:
    """True for dot-prefixed names (``.gitkeep``, ``.DS_Store``)."""
    return name.startswith(".")


def _is_contained(child: Path, parent: Path) -> bool:
    """True when ``child`` resolves to a location inside ``parent``.

    Uses ``relative_to`` on *resolved* paths, which is why it correctly
    rejects a symlink that escapes: resolution follows the link first.
    """
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def discover(
    root: str | Path,
    *,
    base_dir: str | Path | None = None,
    recursive: bool = True,
) -> DiscoveryResult:
    """Find supported and unsupported documents under ``root``.

    Parameters
    ----------
    root:
        Directory to scan. Must exist and be a directory.
    base_dir:
        When given, ``root`` must resolve inside this directory. Pass the
        configured data directory here to enforce "never read outside
        ``DATA_DIR``" at the boundary rather than trusting every caller.
    recursive:
        Descend into sub-directories (default). When ``False``, only the
        top level is scanned.

    Returns
    -------
    DiscoveryResult
        Supported and unsupported files, each with a path proven to be
        inside ``root``.

    Raises
    ------
    NotFoundError
        If ``root`` does not exist, or ``base_dir`` does not contain it.
    ValidationError
        If ``root`` exists but is a file rather than a directory.

    Notes
    -----
    Hidden files and the directories in :data:`IGNORED_DIRECTORY_NAMES` are
    skipped without appearing in either list. Symbolic links are not
    followed for directory traversal, which prevents cycles and escapes.
    """
    root_path = Path(root).expanduser()

    if not root_path.exists():
        raise NotFoundError(
            f"discovery directory does not exist: {root_path.name}",
            code="discovery_root_not_found",
            # Intentionally the basename only: absolute server paths must not
            # escape in an error message a client may see.
            detail=f"resolved to {root_path.resolve()}",
        )

    if not root_path.is_dir():
        raise ValidationError(
            f"discovery root must be a directory, got a file: {root_path.name}",
            code="discovery_root_not_a_directory",
        )

    resolved_root = root_path.resolve()

    if base_dir is not None:
        base_path = Path(base_dir).expanduser()
        if not _is_contained(resolved_root, base_path):
            raise ValidationError(
                "discovery root is outside the configured data directory",
                code="discovery_root_outside_base",
                detail=f"root={resolved_root} base={base_path.resolve()}",
            )

    files: list[DiscoveredFile] = []
    unsupported: list[UnsupportedFile] = []

    # followlinks=False is the containment guarantee: a directory symlink
    # pointing at C:\Users is not descended into, so it cannot be indexed.
    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        current = Path(dirpath)

        # Prune ignored and hidden directories in place (os.walk honours this).
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _is_hidden(name) and name not in IGNORED_DIRECTORY_NAMES
        )

        if not recursive and current != resolved_root:
            dirnames[:] = []
            continue

        for filename in sorted(filenames):
            if _is_hidden(filename):
                continue

            candidate = current / filename

            # A symlinked *file* can still point outside the root even though
            # we never follow directory links, so check containment explicitly.
            if not _is_contained(candidate, resolved_root):
                unsupported.append(
                    UnsupportedFile(
                        relative_path=_relative_posix(candidate, resolved_root),
                        filename=filename,
                        suffix=candidate.suffix.lower(),
                        reason="symlink resolves outside the discovery directory",
                    )
                )
                continue

            relative_path = _relative_posix(candidate, resolved_root)
            suffix = candidate.suffix.lower()
            media_type = EXTENSION_TO_MEDIA_TYPE.get(suffix)

            if media_type is None:
                unsupported.append(
                    UnsupportedFile(
                        relative_path=relative_path,
                        filename=filename,
                        suffix=suffix,
                        reason=_unsupported_reason(suffix),
                    )
                )
                continue

            try:
                size_bytes = candidate.stat().st_size
            except OSError as exc:  # pragma: no cover - race: file vanished
                unsupported.append(
                    UnsupportedFile(
                        relative_path=relative_path,
                        filename=filename,
                        suffix=suffix,
                        reason=f"could not read file metadata: {exc.__class__.__name__}",
                    )
                )
                continue

            files.append(
                DiscoveredFile(
                    path=candidate,
                    relative_path=relative_path,
                    filename=filename,
                    media_type=media_type,
                    size_bytes=size_bytes,
                )
            )

    return DiscoveryResult(
        root=resolved_root,
        files=tuple(files),
        unsupported=tuple(unsupported),
    )


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` using forward slashes.

    Forward slashes make the value stable across Windows and POSIX, which
    matters because :func:`app.domain.models.make_doc_id` hashes this string
    — the same file must yield the same id on either platform.
    """
    return PurePosixPath(path.relative_to(root)).as_posix()


def _unsupported_reason(suffix: str) -> str:
    """Explain why a file was skipped, in terms a user can act on."""
    if not suffix:
        return "file has no extension, so its format cannot be determined"
    supported = ", ".join(sorted(EXTENSION_TO_MEDIA_TYPE))
    return f"extension {suffix!r} is not supported (supported: {supported})"


def describe(result: DiscoveryResult) -> dict[str, int | str]:
    """Return a JSON-safe summary suitable for logs and API responses.

    Contains counts and the root *name* only — never absolute paths or
    file contents.
    """
    return {
        "root": result.root.name,
        "supported": result.supported_count,
        "unsupported": result.unsupported_count,
        "total_size_bytes": result.total_size_bytes,
        "is_empty": result.is_empty,
    }


__all__ = [
    "IGNORED_DIRECTORY_NAMES",
    "DiscoveredFile",
    "DiscoveryResult",
    "UnsupportedFile",
    "describe",
    "discover",
]
