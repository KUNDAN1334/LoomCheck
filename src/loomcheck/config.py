"""Environment loading and model pricing.

A missing setting fails here, at startup, rather than at turn 40 of a suite run when the
agent first reaches for it. There is no defaulting: a wrong database quietly accepted is
worse than a run that refuses to start.

Settings are split by what the caller needs. `alembic upgrade head` requires a database and
nothing else, so requiring an API key to run a migration would be a lie about what the
command does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Repository root. Scenario files name their fixtures relative to this, so that a scenario
reads the same whether it was loaded by path, by directory, or by a test."""


class ConfigError(Exception):
    """A required environment variable is missing, or a model has no price."""


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. See PRICES for where these came from."""

    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    # Groq list pricing, read 2026-08-25. Groq's own pricing page renders client-side and could
    # not be fetched, so these came from third-party model directories and are worth re-checking
    # against a real invoice before quoting cost figures to anyone.
    #
    # Which of these a given API key can actually reach varies by account — some accounts get no
    # llama chat models at all — so `preflight` in runner.py asks the provider rather than
    # assuming this table is the constraint. A model that is reachable but missing here is
    # refused rather than costed at zero, which is why the table is short: a price nobody can
    # source is worse than a model nobody can run.
    #
    # On the free tier these figures are not a bill. They are what the suite *would* cost in
    # production, which is the number a cost regression is about either way.
    "llama-3.3-70b-versatile": ModelPrice(input_per_mtok=0.59, output_per_mtok=0.79),
    "llama-3.1-8b-instant": ModelPrice(input_per_mtok=0.05, output_per_mtok=0.08),
    "openai/gpt-oss-120b": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
    "openai/gpt-oss-20b": ModelPrice(input_per_mtok=0.075, output_per_mtok=0.30),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Price one model call.

    Raises on an unknown model rather than returning 0.0. A suite that silently reports a
    cost of nothing is worse than one that refuses to run: the budget grader would pass
    every scenario and the cost column in a diff would read as a large improvement.
    """
    price = PRICES.get(model)
    if price is None:
        known = ", ".join(sorted(PRICES)) or "none"
        raise ConfigError(
            f"no price recorded for model {model!r}; add it to PRICES in config.py "
            f"(known models: {known})"
        )
    return (tokens_in * price.input_per_mtok + tokens_out * price.output_per_mtok) / 1_000_000


@dataclass(frozen=True)
class DatabaseSettings:
    url: str


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    model: str


def _env(name: str, hint: str, env_file: Path | None = None) -> str:
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"{name} is not set. {hint}")
    return value


def load_database_settings(env_file: Path | None = None) -> DatabaseSettings:
    return DatabaseSettings(
        url=_env(
            "DATABASE_URL",
            "Copy .env.example to .env, then `docker compose up -d db`.",
            env_file,
        )
    )


def load_llm_settings(env_file: Path | None = None) -> LLMSettings:
    return LLMSettings(
        api_key=_env("GROQ_API_KEY", "Get one at https://console.groq.com/keys.", env_file),
        model=_env(
            "LOOMCHECK_MODEL",
            "Set it to a model listed in PRICES, e.g. llama-3.3-70b-versatile.",
            env_file,
        ),
    )
