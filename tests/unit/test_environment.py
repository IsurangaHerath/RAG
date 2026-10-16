"""Environment tests for Stage 05.

These tests answer one question: *can a fresh clone of this repository
actually run?*  They check that every direct dependency listed in
``requirements.txt`` is installed and importable, and that the specific
classes later stages rely on still exist under their expected names.

LangChain 1.x moved many symbols between sub-packages, so asserting the
import paths here catches an upgrade breaking the project before any
pipeline code does.

Nothing in this module contacts the network or needs an API key.
"""

from __future__ import annotations

import importlib
from importlib import metadata

import pytest

# distribution name -> importable module name (they differ for scikit-learn).
REQUIRED_PACKAGES: dict[str, str] = {
    "langchain": "langchain",
    "langchain-core": "langchain_core",
    "langchain-openai": "langchain_openai",
    "langchain-text-splitters": "langchain_text_splitters",
    "langchain-chroma": "langchain_chroma",
    "openai": "openai",
    "chromadb": "chromadb",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "pytest": "pytest",
}


def _requirements_txt() -> str:
    """Return the contents of ``requirements.txt`` from the repository root."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return (root / "requirements.txt").read_text(encoding="utf-8")


def test_requirements_txt_pins_every_dependency() -> None:
    """Every non-comment line in requirements.txt uses an exact ``==`` pin.

    Unpinned dependencies silently drift, which makes a build from two
    weeks ago unreproducible.
    """
    lines = [
        line.strip()
        for line in _requirements_txt().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines, "requirements.txt has no dependencies"

    unpinned = [line for line in lines if "==" not in line]
    assert not unpinned, (
        "Unpinned dependencies found (use ==<version> so installs reproduce): "
        + ", ".join(unpinned)
    )


@pytest.mark.parametrize("distribution", REQUIRED_PACKAGES, ids=REQUIRED_PACKAGES.keys())
def test_direct_dependency_is_installed(distribution: str) -> None:
    """Each declared dependency is installed in the active environment."""
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - env failure
        raise AssertionError(
            f"{distribution!r} is missing. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    assert version, f"{distribution} reported an empty version"


@pytest.mark.parametrize("distribution", REQUIRED_PACKAGES, ids=REQUIRED_PACKAGES.keys())
def test_direct_dependency_imports(distribution: str) -> None:
    """Each declared dependency imports without raising."""
    module_name = REQUIRED_PACKAGES[distribution]
    module = importlib.import_module(module_name)

    assert module is not None, f"{module_name} imported as None"


def test_requirements_txt_matches_required_packages() -> None:
    """requirements.txt declares exactly the packages these tests expect.

    Guards against someone deleting a dependency from requirements.txt
    while it is still required (or adding an undeclared one).
    """
    declared = {
        line.split("==")[0].split("[")[0].strip()
        for line in _requirements_txt().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert declared == set(REQUIRED_PACKAGES), (
        "requirements.txt and REQUIRED_PACKAGES have drifted apart. "
        f"Only in requirements.txt: {sorted(declared - set(REQUIRED_PACKAGES))}; "
        f"only in tests: {sorted(set(REQUIRED_PACKAGES) - declared)}"
    )


# --- Symbols later stages depend on ---------------------------------------
# Import paths are asserted explicitly because LangChain 1.x relocated
# several classes that lived in `langchain` during 0.x.


@pytest.mark.parametrize(
    ("module_path", "symbol"),
    [
        ("langchain_openai", "ChatOpenAI"),
        ("langchain_openai", "OpenAIEmbeddings"),
        ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
        ("langchain_chroma", "Chroma"),
        ("langchain_core.documents", "Document"),
        ("langchain_core.prompts", "ChatPromptTemplate"),
        ("langchain_core.output_parsers", "StrOutputParser"),
        ("fastapi", "FastAPI"),
        ("pydantic", "BaseModel"),
    ],
)
def test_expected_symbols_are_importable(module_path: str, symbol: str) -> None:
    """The exact classes used by later stages still exist at these paths."""
    module = importlib.import_module(module_path)

    assert hasattr(module, symbol), (
        f"{module_path}.{symbol} not found. If LangChain moved it, update the "
        "import in the application module and this test together."
    )


def test_recursive_splitter_produces_distinct_chunks() -> None:
    """Chunking works offline with the installed LangChain version.

    This is a real behavioural check (not just an import check) that runs
    without any API key.

    The input uses numbered, unique tokens on purpose.  Repeating an
    identical short token would make every chunk land on the same relative
    offset, producing byte-identical chunks -- which tests the input's
    periodicity rather than the splitter.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = " ".join(f"token{i:03d}" for i in range(80))
    chunks = splitter.split_text(text)

    assert len(chunks) > 1, "splitter did not split the text at all"
    assert all(chunk.strip() for chunk in chunks), "splitter produced an empty chunk"
    assert all(len(chunk) <= 60 for chunk in chunks), (
        "chunks exceed chunk_size plus a reasonable allowance"
    )
    assert len(set(chunks)) > 1, (
        "all chunks are byte-identical; splitting is not actually happening"
    )
    # Overlap means consecutive chunks share a boundary token.
    assert chunks[1].split()[0] in chunks[0], (
        "expected chunk overlap, but consecutive chunks share no tokens"
    )


def test_chroma_ephemeral_client_can_be_created() -> None:
    """ChromaDB works locally without network access or an API key."""
    import chromadb

    client = chromadb.EphemeralClient()

    assert client is not None
