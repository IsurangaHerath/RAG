"""Guard-rail tests for the project's folder structure (Stage 04).

These are deliberately simple: they fail if a package is accidentally
deleted, renamed, or left without an ``__init__.py``.  That matters because
every later stage drops modules into these folders.

No third-party packages are imported here, so this test runs even before
the virtual environment is installed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Repository root (this file lives in <root>/tests/unit/).
ROOT = Path(__file__).resolve().parents[2]

# Every package that must exist, with a short reason so a failure is
# immediately understandable to a newcomer.
REQUIRED_PACKAGES: dict[str, str] = {
    "app": "top-level application package",
    "app.domain": "shared data models and errors",
    "app.ingestion": "file discovery, loaders, metadata, dedup, reporting",
    "app.preprocessing": "chunking",
    "app.embeddings": "embedding interface and OpenAI adapter",
    "app.storage": "ChromaDB vector store and document registry",
    "app.retrieval": "semantic retrieval",
    "app.generation": "prompting and LLM answer generation",
    "app.evaluation": "metrics and experiment runners",
    "app.api": "FastAPI app wiring and schemas",
    "app.api.routes": "HTTP route handlers",
    "app.observability": "logging and metrics",
    "tests": "test suite root",
    "tests.unit": "offline mocked tests",
    "tests.integration": "real local component tests",
    "scripts": "runnable helper scripts (kept as a package so tests can reuse helpers)",
}

# Directories that hold data rather than code, so they use a .gitkeep
# instead of an __init__.py (git cannot track a truly empty folder).
REQUIRED_DATA_DIRS: dict[str, str] = {
    "data": "local data root (git-ignored, except .gitkeep)",
    "docs": "generated reports and notes",
}


@pytest.mark.parametrize("package", REQUIRED_PACKAGES, ids=REQUIRED_PACKAGES.keys())
def test_package_exists_with_init_file(package: str) -> None:
    """Each expected package exists on disk and declares ``__init__.py``."""
    package_dir = ROOT / Path(*package.split("."))
    reason = REQUIRED_PACKAGES[package]

    assert package_dir.is_dir(), f"Missing package directory for {package!r}: {reason}"
    assert (package_dir / "__init__.py").is_file(), (
        f"{package_dir} is missing __init__.py; Python would not import it as a package."
    )


@pytest.mark.parametrize("package", REQUIRED_PACKAGES, ids=REQUIRED_PACKAGES.keys())
def test_package_is_importable(package: str) -> None:
    """Each package imports cleanly without side effects or extra dependencies."""
    module = importlib.import_module(package)

    assert module is not None
    assert getattr(module, "__doc__", None), (
        f"{package} has no module docstring; every package should say what it is for."
    )


@pytest.mark.parametrize("dirname", REQUIRED_DATA_DIRS, ids=REQUIRED_DATA_DIRS.keys())
def test_data_dir_exists_with_gitkeep(dirname: str) -> None:
    """Non-code directories exist and are trackable by git via a .gitkeep."""
    directory = ROOT / dirname

    assert directory.is_dir(), f"Missing {dirname!r}: {REQUIRED_DATA_DIRS[dirname]}"
    assert (directory / ".gitkeep").exists(), (
        f"{directory} needs a .gitkeep so git does not drop the empty folder."
    )


@pytest.mark.parametrize("dirname", ["data", "docs"], ids=["data", "docs"])
def test_data_dirs_are_not_python_packages(dirname: str) -> None:
    """``data/`` and ``docs/`` hold files, not code, so they must not be packages.

    A stray ``__init__.py`` there would invite accidental imports of uploaded
    documents or generated reports.
    """
    package_marker = ROOT / dirname / "__init__.py"

    assert not package_marker.exists(), (
        f"{package_marker} should not exist; {dirname}/ is a data directory, not a package."
    )


def test_no_module_is_empty_placeholder() -> None:
    """Guard against premature abstraction: only ``__init__.py`` files ship in Stage 04.

    Later stages add the real modules.  Keeping this assertion means the
    scaffold cannot silently fill up with empty placeholder files.
    """
    python_files = [
        path
        for path in (ROOT / "app").rglob("*.py")
        if path.name != "__init__.py"
    ]

    assert not python_files, (
        "Stage 04 must not add implementation modules yet; found: "
        + ", ".join(str(p.relative_to(ROOT)) for p in python_files)
    )


def test_no_secret_files_are_committed() -> None:
    """A ``.env`` file must never be added to the repository."""
    tracked_env = ROOT / ".env"
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert not tracked_env.exists() or ".env" in gitignore, (
        ".env exists but is not covered by .gitignore"
    )
    assert ".env" in gitignore and ".env.example" in gitignore, (
        ".gitignore should ignore .env but keep .env.example"
    )
