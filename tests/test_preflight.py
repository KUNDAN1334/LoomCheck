"""Preflight tests.

The check exists because a model an account cannot reach used to surface as a sixty-line HTTP
traceback on the first scenario, after the suite had already started spending. These pin that it
now fails as a configuration error, before anything is spent, naming the fix.

The provider is stubbed rather than called — the point is the message, not the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest import MonkeyPatch

from loomcheck import runner
from loomcheck.config import ConfigError, LLMSettings


@dataclass(frozen=True)
class FakeModel:
    id: str


class FakeListing:
    def __init__(self, ids: list[str]) -> None:
        self.data = [FakeModel(i) for i in ids]


def stub_smoke_call(monkeypatch: MonkeyPatch, error: Exception | None = None) -> None:
    """Stand in for the one real call preflight makes with the tools bound."""

    class FakeBound:
        def invoke(self, messages: object) -> str:
            if error is not None:
                raise error
            return "ready"

    class FakeModel:
        def bind_tools(self, tools: object, **kwargs: object) -> FakeBound:
            return FakeBound()

    monkeypatch.setattr(runner, "make_model", lambda settings: FakeModel())


def stub_groq(monkeypatch: MonkeyPatch, ids: list[str]) -> None:
    class FakeModels:
        def list(self) -> FakeListing:
            return FakeListing(ids)

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(runner, "Groq", FakeGroq)


def settings(model: str) -> LLMSettings:
    return LLMSettings(api_key="gsk_fake", model=model)


def test_a_model_the_key_cannot_reach_lists_the_ones_it_can(monkeypatch: MonkeyPatch) -> None:
    stub_groq(monkeypatch, ["llama-3.1-8b-instant", "openai/gpt-oss-20b"])
    stub_smoke_call(monkeypatch)
    with pytest.raises(ConfigError) as exc:
        runner.preflight(settings("llama-3.3-70b-versatile"))

    message = str(exc.value)
    assert "'llama-3.3-70b-versatile', which this API key cannot use" in message
    assert "llama-3.1-8b-instant" in message
    assert "openai/gpt-oss-20b" in message
    assert "LOOMCHECK_MODEL in .env" in message


def test_an_available_but_unpriced_model_is_refused_rather_than_costed_at_zero(
    monkeypatch: MonkeyPatch,
) -> None:
    """Running it would report a suite that cost nothing, which the budget grader would pass and
    the next diff would show as a large improvement."""
    stub_groq(monkeypatch, ["some-new-model"])
    stub_smoke_call(monkeypatch)
    with pytest.raises(ConfigError) as exc:
        runner.preflight(settings("some-new-model"))

    assert "has no price recorded" in str(exc.value)
    assert "PRICES in config.py" in str(exc.value)


def test_a_model_that_is_reachable_priced_and_takes_tools_passes(monkeypatch: MonkeyPatch) -> None:
    stub_groq(monkeypatch, ["llama-3.1-8b-instant"])
    stub_smoke_call(monkeypatch)
    runner.preflight(settings("llama-3.1-8b-instant"))


def test_a_model_that_refuses_tool_calling_is_caught_before_the_suite_starts(
    monkeypatch: MonkeyPatch,
) -> None:
    """Half the models on a Groq account are transcription, speech or classifier models. They
    are listed, they answer chat, and they reject tools — which the listing does not say."""
    from groq import GroqError

    stub_groq(monkeypatch, ["whisper-large-v3", "llama-3.1-8b-instant"])
    stub_smoke_call(monkeypatch, GroqError("`tool calling` is not supported with this model"))

    with pytest.raises(ConfigError) as exc:
        runner.preflight(settings("llama-3.1-8b-instant"))

    message = str(exc.value)
    assert "will not serve the request this harness makes" in message
    assert "tool calling" in message


def test_a_model_needing_terms_acceptance_is_caught_the_same_way(
    monkeypatch: MonkeyPatch,
) -> None:
    from groq import GroqError

    stub_groq(monkeypatch, ["llama-3.1-8b-instant"])
    stub_smoke_call(monkeypatch, GroqError("requires terms acceptance"))

    with pytest.raises(ConfigError, match="requires terms acceptance"):
        runner.preflight(settings("llama-3.1-8b-instant"))


def test_an_unreachable_provider_is_a_config_error_not_a_traceback(
    monkeypatch: MonkeyPatch,
) -> None:
    from groq import GroqError

    class ExplodingGroq:
        def __init__(self, api_key: str) -> None:
            raise GroqError("bad key")

    monkeypatch.setattr(runner, "Groq", ExplodingGroq)
    with pytest.raises(ConfigError, match="could not ask Groq"):
        runner.preflight(settings("llama-3.1-8b-instant"))
