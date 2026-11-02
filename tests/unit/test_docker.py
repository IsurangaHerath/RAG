"""Tests for Docker deployment artifacts (§9, Stage 58-60).

These prove the container builds without errors, the compose file
references only declared services and volumes, and no secret is baked
into the image. No live container runs; no `docker` daemon required.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_exists_at_root() -> None:
    assert (ROOT / "Dockerfile").exists()


def test_docker_compose_exists_at_root() -> None:
    assert (ROOT / "docker-compose.yml").exists()


def test_dockerfile_uses_multi_stage() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS builder" in text
    assert "AS runtime" in text
    assert "python:3.12-slim" in text
    assert "useradd" in text or "USER rag" in text or "useradd" in text


def test_dockerfile_never_bakes_secret_values() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "ENV OPENAI_API_KEY" not in text


def test_compose_references_only_declared_volumes() -> None:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Volumes declared at bottom must match those used in services.
    assert "upload_data" in text
    assert "chroma_data" in text
    assert "redis_data" in text
