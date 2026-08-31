"""The tools the agent under test is shown.

Schemas only — there are no function bodies here, because there is no claims system behind
this. Nothing is ever "swapped" for a mock: the agent's single route to the outside world is
the resolver, and that is the whole of the interception design. See docs/flow.md.

Each tool is a name, a description saying *when* to use it, and a typed input schema. That is
deliberately the shape a Custom Tool takes on the platform this is aimed at, so a real tool
definition can be pointed at this harness by writing a scenario rather than by writing an
adapter.

The split is deliberate too. `policy_lookup`, `precedent_search` and `claim_history` only read,
which mirrors the read-only baseline every new agent starts from. The four terminal tools are
the ones that have an effect, and they are the ones that would have to be granted explicitly.

Terminal tools never reach the resolver, because calling one ends the stage. Each takes a
reason, since the procedure asks the agent to state the fact its decision rests on and there
would otherwise be nowhere to put it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from loomcheck.models import Outcome


class PolicyLookupArgs(BaseModel):
    policy_number: str = Field(description="The policy number quoted in the claim, e.g. NF-7710")


class PrecedentSearchArgs(BaseModel):
    query: str = Field(description="Describe the loss in a sentence: peril, cause, property type")


class ClaimHistoryArgs(BaseModel):
    policy_number: str = Field(description="The policy number quoted in the claim")


class ApproveClaimArgs(BaseModel):
    amount_eur: float = Field(description="Amount to approve in euros, before excess")
    reason: str = Field(description="The fact this approval rests on, in one sentence")


class DenyClaimArgs(BaseModel):
    reason: str = Field(description="The policy term or fact this denial rests on")


class RequestMoreInfoArgs(BaseModel):
    question: str = Field(description="What you need from the claimant, and only they can give")


class BlockedArgs(BaseModel):
    reason: str = Field(description="What a senior handler needs to decide, and why you cannot")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "policy_lookup",
        "Look up a policy: status, cover, limits, excess and exclusions.",
        PolicyLookupArgs,
    ),
    ToolSpec(
        "precedent_search",
        "Search past cases for how the company has handled a loss like this one.",
        PrecedentSearchArgs,
    ),
    ToolSpec(
        "claim_history",
        "List prior claims on a policy, with any payment or conduct flags.",
        ClaimHistoryArgs,
    ),
    ToolSpec(
        "approve_claim",
        "Approve the claim. Use when the file supports it. Ends the stage.",
        ApproveClaimArgs,
    ),
    ToolSpec(
        "deny_claim",
        "Deny the claim. Use when a term or fact excludes it. Ends the stage.",
        DenyClaimArgs,
    ),
    ToolSpec(
        "request_more_info",
        "Ask the claimant for something only they can supply. Use when the missing fact is "
        "theirs to give. Ends the stage; it waits on their reply.",
        RequestMoreInfoArgs,
    ),
    ToolSpec(
        "blocked",
        "Halt the case and wait for a senior handler. Use when the decision is not yours to "
        "make. Ends the stage; a person has to unblock it.",
        BlockedArgs,
    ),
)

TERMINAL_OUTCOMES: dict[str, Outcome] = {
    "approve_claim": Outcome.APPROVE_CLAIM,
    "deny_claim": Outcome.DENY_CLAIM,
    "request_more_info": Outcome.REQUEST_MORE_INFO,
    "blocked": Outcome.BLOCKED,
}


def tool_schemas() -> list[dict[str, Any]]:
    """The tool list in the shape `bind_tools` wants, with JSON schema from Pydantic.

    Written as dicts rather than passing the Pydantic classes directly because LangChain
    derives a tool's name from the class name, and these names have to match the keys a
    scenario author writes in a YAML `tools:` block.
    """
    schemas: list[dict[str, Any]] = []
    for spec in TOOLS:
        parameters = spec.args.model_json_schema()
        parameters.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                },
            }
        )
    return schemas
