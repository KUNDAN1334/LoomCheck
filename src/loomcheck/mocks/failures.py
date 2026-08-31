"""What an injected tool failure looks like to the agent.

The four modes are not four flavours of the same thing. They differ in how *detectable* the
failure is, which is the whole point — an agent's recovery behaviour is only interesting when
noticing is the hard part.

    server_error    loud   the tool channel reports an error; impossible to miss
    timeout         loud   same, with a different cause the agent might treat differently
    malformed_json  noisy  the call "succeeded" and returned unparseable garbage
    empty_result    quiet  the call succeeded, the JSON parses, and there is nothing in it

`empty_result` is the one worth building the suite around. It is indistinguishable from a
legitimate no-results answer unless the agent thinks about whether no-results is plausible
here, which is the same judgement the escalation grader measures from the other side.

Payloads are fixed strings, not generated: a failure that differed between runs would make the
harness the source of the non-determinism it exists to detect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loomcheck.models import InjectedFailure


@dataclass(frozen=True)
class FailureResponse:
    """What comes back, and whether the tool channel itself admits something went wrong."""

    payload: str
    is_error: bool


TRUNCATED_JSON = '{"policy_number": "NF-8842", "status": "act'
"""Cut off mid-value, the way a response dropped by a proxy actually arrives — not a string of
nonsense. An agent that parses defensively should notice; one that string-matches may not."""


def render_failure(failure: InjectedFailure, tool: str) -> FailureResponse:
    """Render one failure mode for one tool. Deterministic, and the same on every machine."""
    if failure is InjectedFailure.SERVER_ERROR:
        return FailureResponse(
            payload=json.dumps(
                {"error": "internal_server_error", "status": 500, "tool": tool},
            ),
            is_error=True,
        )

    if failure is InjectedFailure.TIMEOUT:
        return FailureResponse(
            payload=json.dumps(
                {"error": "timeout", "tool": tool, "detail": "no response after 30s"},
            ),
            is_error=True,
        )

    if failure is InjectedFailure.MALFORMED_JSON:
        # Not an error on the channel: as far as the transport is concerned this call worked.
        return FailureResponse(payload=TRUNCATED_JSON, is_error=False)

    # empty_result: well-formed, parseable, and says nothing.
    return FailureResponse(payload=json.dumps({}), is_error=False)
