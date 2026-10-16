"""README accuracy tests (Stage 08).

A README that overstates progress is worse than none: it misleads users and
recruiters, and it is never re-read after launch. These tests pin the few
claims that can silently rot:

* links must resolve to files that exist,
* documented configuration defaults must match the real `Settings`,
* the status section must keep admitting the pipeline is not finished until
  that actually changes.

They are intentionally cheap and offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Read the README once for the whole module."""
    assert README.is_file(), "README.md is missing"
    return README.read_text(encoding="utf-8")


def test_readme_exists_and_is_non_trivial(readme_text: str) -> None:
    """The README exists and is substantial enough to be useful."""
    assert len(readme_text) > 1500, "README.md looks too short to be a real README"


def test_readme_local_links_resolve(readme_text: str) -> None:
    """Every relative Markdown link points at a file that exists.

    Catches a renamed or deleted file leaving a dead link behind.
    """
    import re

    targets = {
        match.group(1) for match in re.finditer(r"\]\((?!https?://|#)([^)#]+)\)", readme_text)
    }
    assert targets, "expected at least one local link"

    missing = [t for t in sorted(targets) if not (ROOT / t).exists()]
    assert not missing, f"README links to missing files: {missing}"


def test_readme_does_not_claim_the_pipeline_works(readme_text: str) -> None:
    """Until it is genuinely finished, the README must not imply a working pipeline.

    This is the single most important honesty guard in Stage 08. Phrases
    below are the ones that would misrepresent current progress.
    """
    lowered = readme_text.lower()

    forbidden = [
        "the pipeline works",
        "pipeline is complete",
        "fully functional",
        "production ready",
        "production-ready",
        "achieves %",  # no measured accuracy exists yet
        "% accuracy",
        "outperforms",
    ]
    found = [phrase for phrase in forbidden if phrase in lowered]

    assert not found, "README makes claims that are not yet verified by measurement: " + ", ".join(
        found
    )


def test_readme_marks_unimplemented_features_as_planned(readme_text: str) -> None:
    """Features that do not exist yet are labelled planned, not promised."""
    assert "planned" in readme_text.lower(), (
        "README must clearly label unimplemented features as planned"
    )
    assert "not yet implemented" in readme_text.lower() or "not yet" in readme_text.lower(), (
        "README must state plainly what has not been built"
    )


def test_readme_admits_no_benchmarks_yet(readme_text: str) -> None:
    """No performance numbers are presented as results."""
    assert "not yet measured" in readme_text.lower(), (
        "README should say explicitly that performance has not been measured"
    )


def test_readme_config_defaults_match_settings(readme_text: str) -> None:
    """Every default documented in the README matches `Settings` exactly.

    Documents drift from code constantly; this makes drift a test failure.
    """
    settings = Settings(_env_file=None)

    documented_defaults = {
        "CHAT_MODEL": settings.chat_model,
        "EMBEDDING_MODEL": settings.embedding_model,
        "CHUNK_SIZE": settings.chunk_size,
        "CHUNK_OVERLAP": settings.chunk_overlap,
        "TOP_K": settings.top_k,
        "MAX_UPLOAD_MB": settings.max_upload_mb,
        "MAX_QUERY_CHARS": settings.max_query_chars,
        "LOG_LEVEL": settings.log_level,
    }

    for var, actual in documented_defaults.items():
        assert f"`{var}`" in readme_text, f"{var} is not documented in the README"
        # The README shows the value in backticks in the config table.
        assert f"`{actual}`" in readme_text, (
            f"README does not document the real default for {var}: {actual!r}"
        )


def test_readme_lists_actual_direct_dependency_count(readme_text: str) -> None:
    """The dependency count stated in the README matches requirements.txt."""
    import re

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    actual = len(
        [
            line
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    )

    match = re.search(r"(\d+)\s+pinned direct dependencies", readme_text)
    assert match, "README should state how many direct dependencies are pinned"
    assert int(match.group(1)) == actual, (
        f"README claims {match.group(1)} pinned dependencies, requirements.txt has {actual}"
    )


def test_readme_shows_test_count_matching_reality(readme_text: str) -> None:
    """The advertised test count is plausible for the current suite.

    We do not hard-assert an exact number (it changes every stage), but the
    figure must be a real, non-zero count and not larger than what the
    suite could possibly contain.
    """
    import re

    match = re.search(r"\*\*(\d+) tests? passing\*\*", readme_text)
    assert match, "README should state the number of passing tests"

    claimed = int(match.group(1))
    assert claimed > 0
    # Sanity ceiling: no suite may claim more tests than files could hold.
    test_files = list((ROOT / "tests").rglob("test_*.py"))
    assert claimed <= len(test_files) * 200, (
        f"README claims {claimed} tests, which looks inflated for {len(test_files)} test files"
    )
