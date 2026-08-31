"""The agent under test: a small ReAct loop over the seven claim tools.

Hand-rolled rather than `create_react_agent`, because the prebuilt agent owns the loop and
binds its tools at construction, which puts the recorder outside the thing it needs to watch.
Two nodes and two routing functions is the whole of it.

Nothing here knows it is being tested. There is no branch for mocks: `call_tools` asks the
resolver because the resolver is the only source of tool results that exists.
"""

from __future__ import annotations

from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from loomcheck.agent.tools import TERMINAL_OUTCOMES, tool_schemas
from loomcheck.mocks.resolver import MockResolver
from loomcheck.models import Outcome
from loomcheck.recorder import Recorder


class ParallelToolCallError(Exception):
    """The model emitted more than one tool call in a single response."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    outcome: Outcome | None
    turns: int


def build_agent(
    model: BaseChatModel,
    resolver: MockResolver,
    recorder: Recorder,
    max_turns: int,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile a graph bound to one scenario's resolver and recorder.

    Built per scenario rather than once per run: the resolver counts calls per tool, so
    sharing one across scenarios would let the second scenario in a suite see the first
    scenario's call history.
    """
    # `tool_choice="required"` states the contract at the binding: the agent's only route out of
    # this loop is a tool call, so every turn must be one. Without it, gpt-oss-class models end
    # the case by *writing* `deny_claim({...})` into the message content — the right decision in
    # the wrong form — and the harness scores a judgement failure that was a protocol failure.
    # See docs/learning.md L-017.
    bound = model.bind_tools(tool_schemas(), tool_choice="required", parallel_tool_calls=False)

    def call_model(state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        response = bound.invoke(state["messages"])
        latency_ms = int((perf_counter() - started) * 1000)
        usage = getattr(response, "usage_metadata", None) or {}
        recorder.observe_model_call(
            latency_ms=latency_ms,
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
            text=str(getattr(response, "text", "") or ""),
        )
        return {"messages": [response], "turns": state["turns"] + 1}

    def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        if len(last.tool_calls) > 1:
            # Parallel calls are disabled at bind time. If one arrives anyway, failing is the
            # only honest option: recording the first and dropping the rest would produce a
            # trace that does not describe what the agent did.
            names = ", ".join(call["name"] for call in last.tool_calls)
            raise ParallelToolCallError(
                f"model emitted {len(last.tool_calls)} tool calls in one turn ({names}); "
                "loomcheck records one tool call per turn"
            )

        call = last.tool_calls[0]
        name, arguments = call["name"], dict(call["args"])

        if name in TERMINAL_OUTCOMES:
            recorder.capture(name, arguments, "case resolved")
            return {"outcome": TERMINAL_OUTCOMES[name]}

        resolution = resolver.resolve(name)
        recorder.capture(name, arguments, resolution.payload, resolution.injected_failure)
        return {
            "messages": [
                ToolMessage(
                    content=resolution.payload,
                    tool_call_id=call["id"] or name,
                    # A 500 arrives flagged; a response that parses cleanly and says nothing
                    # does not. Flattening the two would hand the agent a signal it would not
                    # have in production, and recovery would grade an easier problem.
                    status="error" if resolution.is_error else "success",
                )
            ]
        }

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if has_calls else END

    def after_tools(state: AgentState) -> str:
        if state["outcome"] is not None:
            return END
        return END if state["turns"] >= max_turns else "model"

    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    return graph.compile()


def initial_state(system_prompt: str, case: str) -> AgentState:
    """The agent sees the procedure and the inbound case. It never sees ground_truth."""
    return AgentState(
        messages=[SystemMessage(content=system_prompt), HumanMessage(content=case)],
        outcome=None,
        turns=0,
    )
