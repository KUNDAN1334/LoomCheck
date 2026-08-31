"""Trace capture.

A turn is one LLM call plus the single tool call it produced. That pairing is what makes
`max_tool_calls` and `max_cost_usd` count the same units, and it is why the recorder is a
separate object from the runner: the runner drives the loop, the recorder decides what a
turn is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loomcheck.config import cost_usd
from loomcheck.models import InjectedFailure, Outcome, ScenarioResult, TurnRecord


@dataclass(frozen=True)
class ModelCall:
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class Recorder:
    """Collects one scenario's turns as the graph runs."""

    def __init__(self, scenario_id: str, model: str) -> None:
        self._scenario_id = scenario_id
        self._model = model
        self._turns: list[TurnRecord] = []
        self._pending: ModelCall | None = None
        self._unpaired_cost_usd = 0.0
        self._last_text = ""

    def observe_model_call(
        self, latency_ms: int, tokens_in: int, tokens_out: int, text: str = ""
    ) -> None:
        """Hold one LLM call's metrics until the tool call it produced arrives.

        `text` is kept for the same reason the unpaired cost is: if this call turns out to be
        the last one and produced no tool call, what the agent wrote is the only evidence of
        why the run ended.
        """
        if self._pending is not None:
            # The previous call produced no tool call. Its cost was still real, so it is kept
            # outside the turn list rather than dropped; see `finish`.
            self._unpaired_cost_usd += self._pending.cost_usd
        self._last_text = text
        self._pending = ModelCall(
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd(self._model, tokens_in, tokens_out),
        )

    def capture(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: str,
        injected_failure: InjectedFailure | None = None,
    ) -> None:
        """Pair the held model call with the tool call it emitted, and record the turn."""
        call = self._pending or ModelCall(0, 0, 0, 0.0)
        self._pending = None
        self._turns.append(
            TurnRecord(
                index=len(self._turns) + 1,
                tool=tool,
                arguments=arguments,
                result=result,
                injected_failure=injected_failure,
                latency_ms=call.latency_ms,
                tokens_in=call.tokens_in,
                tokens_out=call.tokens_out,
                cost_usd=call.cost_usd,
            )
        )

    def finish(self, outcome: Outcome | None) -> ScenarioResult:
        """Close the trace.

        `total_cost_usd` is not simply the sum of the turns. A model call that produced no
        tool call — the agent writing prose instead of acting, or hitting the turn ceiling —
        still cost money, and a budget grader that ignored it would let an agent talk itself
        over the ceiling for free.
        """
        unpaired = self._unpaired_cost_usd + (self._pending.cost_usd if self._pending else 0.0)
        return ScenarioResult(
            scenario_id=self._scenario_id,
            outcome=outcome,
            turns=list(self._turns),
            total_cost_usd=sum(turn.cost_usd for turn in self._turns) + unpaired,
            final_message=self._last_text.strip() or None if outcome is None else None,
        )
