"""A scripted chat model.

The test suite never calls an LLM: a regression harness whose own tests are non-deterministic
would be an odd thing to hand someone. This stub plays a fixed list of responses, which lets
every trace the graph can produce — a clean resolution, a runaway loop, an agent that writes
prose instead of acting — be constructed exactly rather than fished for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from groq import BadRequestError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict


def tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    tokens_in: int = 1000,
    tokens_out: int = 40,
) -> AIMessage:
    """One model response that calls one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": f"call_{name}", "type": "tool_call"}],
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    )


def parallel_tool_calls(*names: str) -> AIMessage:
    """A response that calls several tools at once, which the graph must refuse."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {}, "id": f"call_{name}", "type": "tool_call"} for name in names
        ],
        usage_metadata={"input_tokens": 1000, "output_tokens": 40, "total_tokens": 1040},
    )


def prose(text: str = "I think I need to consider this further.") -> AIMessage:
    """A response with no tool call: the agent talking instead of acting."""
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 1000, "output_tokens": 40, "total_tokens": 1040},
    )


def _error(code: str, body_extra: dict[str, Any]) -> BadRequestError:
    body = {
        "error": {
            "message": "bad request",
            "type": "invalid_request_error",
            "code": code,
            **body_extra,
        }
    }
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return BadRequestError("400", response=httpx.Response(400, request=request), body=body)


def rejected(text: str) -> BadRequestError:
    """The 400 a provider returns instead of a response when the model writes prose under
    `tool_choice="required"`. Shaped from a real Groq body, because the harness reads two
    specific keys out of it and a hand-waved stub would not prove they are the right two."""
    return _error("tool_use_failed", {"failed_generation": text})


def other_bad_request() -> BadRequestError:
    """A 400 that is the harness's own fault, which must not be mistaken for agent behaviour."""
    return _error("context_length_exceeded", {})


class ScriptedChatModel(BaseChatModel):
    """Returns queued responses in order, repeating the last one once the script runs out.

    A script entry may be an exception instead of a message, which is how a provider rejection
    gets into a test: the failure this reproduces happens at the call, not in its result.
    """

    # Pydantic fields on BaseChatModel, not class attributes, so the mutable default is fine.
    # `BadRequestError` is not a pydantic model, so the field needs arbitrary types allowed;
    # without it pydantic tries to validate the exception *into* an AIMessage.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: list[BadRequestError | AIMessage] = []  # noqa: RUF012
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        scripted = self.script[index]
        if isinstance(scripted, BadRequestError):
            raise scripted

        # Every response gets a fresh id, as a real provider's would. Returning the same
        # message object twice makes LangGraph's add_messages reducer treat the second as an
        # update to the first and replace it in place, which silently truncates the loop
        # instead of extending it. That is a stub artifact, not agent behaviour.
        response = scripted.model_copy(deep=True)
        response.id = f"stub-{self.calls}"
        for position, call in enumerate(response.tool_calls):
            call["id"] = f"call-{self.calls}-{position}"
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        """Accept and ignore the tool schemas and the binding options. What the agent *may*
        call is the graph's business; what it *does* call is the script's — which is how a
        script can still produce prose even though the real binding requires a tool call."""
        return self
