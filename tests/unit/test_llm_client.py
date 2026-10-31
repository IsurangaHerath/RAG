"""Tests for the `ChatOpenAI` adapter (§5.5, §5.15).

No network: the delegate is either a stub or a fake constructor, so these
tests exercise argument forwarding, response shaping and the translation of
every provider failure family. Credentials must never be copied into the
translated error, which is asserted with a recognisable fake key.
"""

from __future__ import annotations

from typing import Any

import openai
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings, SettingsError
from app.domain.errors import (
    DependencyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)
from app.generation.llm_client import LLMClient
from tests.fakes import openai_connection_error, openai_status_error

FAKE_KEY = "sk-test-DO-NOT-USE-abc123SECRET"


def make_settings(**overrides: object) -> Settings:
    """Build settings that ignore any real ``.env`` on disk."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class StubChat:
    """Minimal ``ChatOpenAI`` delegate: records calls, returns/raises as told."""

    def __init__(self, result: object = None) -> None:
        self.result = result if result is not None else AIMessage(content="answer [1]")
        self.error: BaseException | None = None
        self.calls: list[tuple[list[BaseMessage], dict[str, Any]]] = []

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        self.calls.append((list(messages), kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def test_generate_messages_returns_the_assistant_reply() -> None:
    stub = StubChat(AIMessage(content="It is 2% [1]."))
    client = LLMClient(make_settings(), client=stub)
    messages = [HumanMessage(content="what is the fee?")]

    result = client.generate_messages(messages, temperature=0.2, max_tokens=128)

    assert [message.content for message in result] == ["It is 2% [1]."]
    _, kwargs = stub.calls[0]
    assert kwargs == {"temperature": 0.2, "max_tokens": 128}


def test_a_list_of_messages_is_returned_unchanged() -> None:
    replies = [AIMessage(content="one"), AIMessage(content="two")]
    client = LLMClient(make_settings(), client=StubChat(replies))

    assert client.generate_messages([HumanMessage(content="q")]) == replies


def test_injected_client_needs_no_api_key() -> None:
    client = LLMClient(make_settings(openai_api_key=None), client=StubChat())

    result = client.generate_messages([HumanMessage(content="q")])

    assert result[0].content == "answer [1]"


def test_missing_key_fails_before_a_real_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_construct(*args: object, **kwargs: object) -> None:
        raise AssertionError("the client must not be built without a key")

    monkeypatch.setattr("app.generation.llm_client.ChatOpenAI", must_not_construct)
    client = LLMClient(make_settings(openai_api_key=None))

    with pytest.raises(SettingsError, match="OPENAI_API_KEY is not set"):
        client.generate_messages([HumanMessage(content="q")])


# --- Provider failure translation -----------------------------------------


def test_authentication_failure_never_copies_the_credential() -> None:
    stub = StubChat()
    stub.error = openai_status_error(openai.AuthenticationError, 401, message=f"bad key {FAKE_KEY}")
    client = LLMClient(make_settings(), client=stub)

    with pytest.raises(ProviderAuthenticationError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    error = excinfo.value
    assert error.code == "provider_authentication_failed"
    assert FAKE_KEY not in str(error)
    assert FAKE_KEY not in (error.detail or "")


def test_rate_limit_becomes_the_shared_rate_limit_error() -> None:
    stub = StubChat()
    stub.error = openai_status_error(openai.RateLimitError, 429)
    client = LLMClient(make_settings(), client=stub)

    with pytest.raises(ProviderRateLimitError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.code == "provider_rate_limit"


def test_unexpected_status_becomes_dependency_trouble() -> None:
    stub = StubChat()
    stub.error = openai_status_error(openai.APIStatusError, 500)
    client = LLMClient(make_settings(), client=stub)

    with pytest.raises(DependencyError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.code == "dependency_unavailable"
    assert "500" in (excinfo.value.detail or "")


def test_connection_failure_becomes_dependency_trouble() -> None:
    stub = StubChat()
    stub.error = openai_connection_error()
    client = LLMClient(make_settings(), client=stub)

    with pytest.raises(DependencyError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.code == "dependency_unavailable"
    assert "APIConnectionError" in (excinfo.value.detail or "")


def test_non_provider_failure_still_becomes_dependency_trouble() -> None:
    stub = StubChat()
    stub.error = RuntimeError("boom")
    client = LLMClient(make_settings(), client=stub)

    with pytest.raises(DependencyError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.detail == "RuntimeError"


def test_a_non_message_response_is_rejected() -> None:
    client = LLMClient(make_settings(), client=StubChat(result="plain string"))

    with pytest.raises(DependencyError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.code == "llm_invalid_response"


def test_client_construction_failure_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        raise openai_status_error(openai.AuthenticationError, 401)

    monkeypatch.setattr("app.generation.llm_client.ChatOpenAI", reject)
    client = LLMClient(make_settings(openai_api_key=FAKE_KEY))

    with pytest.raises(ProviderAuthenticationError):
        client.generate_messages([HumanMessage(content="q")])


def test_unexpected_construction_failure_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("bad endpoint")

    monkeypatch.setattr("app.generation.llm_client.ChatOpenAI", reject)
    client = LLMClient(make_settings(openai_api_key=FAKE_KEY))

    with pytest.raises(DependencyError) as excinfo:
        client.generate_messages([HumanMessage(content="q")])

    assert excinfo.value.detail == "ValueError"
