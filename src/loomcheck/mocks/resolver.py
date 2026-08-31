"""The interception seam.

Everything the agent learns about the world comes through `resolve()`. One instance per
scenario run, because a mock with an injected failure answers differently depending on how many
times it has been called — a resolver shared across scenarios would leak that count, and the
second run of a suite would differ from the first.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loomcheck.config import PROJECT_ROOT
from loomcheck.mocks.failures import render_failure
from loomcheck.models import InjectedFailure, Scenario, ToolMock


class UndeclaredToolError(Exception):
    """The agent called an information tool the scenario does not mock."""


@dataclass(frozen=True)
class Resolution:
    """What the agent gets back, plus what the recorder and the agent loop need to know.

    `injected_failure` is for the recorder and, later, the recovery grader. `is_error` is for
    the agent: it decides whether the tool channel admits something went wrong, which is the
    difference between a 500 and a response that parses cleanly and says nothing.
    """

    payload: str
    injected_failure: InjectedFailure | None = None
    is_error: bool = False


class MockResolver:
    """Answers one scenario's tool calls from its fixtures."""

    def __init__(self, scenario: Scenario, root: Path = PROJECT_ROOT) -> None:
        self._scenario_id = scenario.id
        self._mocks = scenario.tools
        self._root = root
        self._calls: Counter[str] = Counter()

    def resolve(self, tool: str) -> Resolution:
        """Return what `tool` produces on this call.

        Arguments are deliberately not a parameter. A scenario fixes one response per tool, so
        what the agent asked for changes the trace but not the answer. Making the response
        depend on arguments would mean scenario authors writing matchers, and a scenario you
        have to debug is a scenario nobody writes.

        The call *count* does change the answer, and only in one way: a mock declaring a
        failure fails its first call and returns `then` on every call after it.
        """
        mock = self._mocks.get(tool)
        if mock is None:
            declared = ", ".join(sorted(self._mocks)) or "none"
            raise UndeclaredToolError(
                f"{self._scenario_id}: the agent called {tool!r}, which this scenario does not "
                f"mock (declared: {declared}). Add it to the scenario's 'tools' block."
            )

        self._calls[tool] += 1

        if mock.failure is not None and self._is_still_failing(tool, mock):
            rendered = render_failure(mock.failure, tool)
            return Resolution(
                payload=rendered.payload,
                injected_failure=mock.failure,
                is_error=rendered.is_error,
            )

        fixture = mock.then if mock.failure is not None else mock.response
        assert fixture is not None  # ToolMock's validator guarantees one of the two paths
        return Resolution(payload=(self._root / fixture).read_text(encoding="utf-8"))

    def _is_still_failing(self, tool: str, mock: ToolMock) -> bool:
        """A mock with no `then` is a permanent outage, which is a legitimate scenario: some
        tools are down for the whole case and the right behaviour is to stop, not to retry
        forever."""
        if mock.then is None:
            return True
        return self._calls[tool] == 1

    def call_count(self, tool: str) -> int:
        return self._calls[tool]
