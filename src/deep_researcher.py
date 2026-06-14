"""Deep research agent: Scope -> Supervisor -> Researcher(s) -> Write."""

from __future__ import annotations

import os
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .configuration import Configuration
from .prompts import (
    BRIEF_PROMPT,
    COMPRESS_PROMPT,
    FINAL_REPORT_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
    today,
)
from .state import AgentState, ResearcherState, SupervisorState
from .utils import (
    get_model,
    get_researcher_tools,
    get_supervisor_tools,
    think_tool,
    web_search,
)


# Minimum gap between model calls to stay under Groq's free-tier rate limit.
_MIN_GAP = float(os.getenv("GROQ_MIN_GAP", "15"))
_last_call = [0.0]


def _throttle(label: str = "") -> None:
    wait = _MIN_GAP - (time.time() - _last_call[0])
    if wait > 0:
        print(f"  [throttle] pausing {wait:.0f}s to respect free-tier limits...")
        time.sleep(wait)
    _last_call[0] = time.time()
    if label:
        print(label)


# --- Delegation tools ---
@tool
def conduct_research(research_topic: str) -> str:
    """Delegate a single, self-contained research topic to a researcher sub-agent."""
    return f"Delegated: {research_topic}"


@tool
def research_complete() -> str:
    """Signal that the current research scope is fully covered."""
    return "Research complete."


# --- 1. Scope ---
def write_research_brief(state: AgentState, config) -> Command:
    cfg = Configuration.from_runnable_config(config)
    model = get_model(cfg.scope_model)
    convo = state["messages"][-1].content[:1500]
    prompt = BRIEF_PROMPT.format(date=today(), messages=convo)
    _throttle("[1/3] SCOPE: writing the research brief...")
    brief = model.invoke([HumanMessage(content=prompt)]).content
    return Command(
        goto="supervisor_subgraph",
        update={
            "research_brief": brief,
            "supervisor_messages": {
                "type": "override",
                "value": [SystemMessage(content=SUPERVISOR_PROMPT.format(
                    date=today(),
                    research_brief=brief,
                    max_concurrent=cfg.max_concurrent_research_units,
                ))],
            },
        },
    )


# --- 2. Supervisor subgraph ---
def supervisor(state: SupervisorState, config) -> Command:
    cfg = Configuration.from_runnable_config(config)
    model = get_model(cfg.research_model).bind_tools(
        get_supervisor_tools(conduct_research, research_complete)
    )
    _throttle("[2/3] RESEARCH: supervisor is planning the next step...")
    response = model.invoke(state["supervisor_messages"])
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


def supervisor_tools(state: SupervisorState, config) -> Command:  # noqa: F811
    cfg = Configuration.from_runnable_config(config)
    last = state["supervisor_messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    # Stop on completion, no tool calls, or iteration cap.
    over_limit = state.get("research_iterations", 0) >= cfg.max_researcher_iterations
    said_done = any(tc["name"] == "research_complete" for tc in tool_calls)
    if not tool_calls or said_done or over_limit:
        return Command(goto=END, update={})

    tool_messages: list[ToolMessage] = []
    new_notes: list[str] = []

    for tc in tool_calls:
        if tc["name"] == "think_tool":
            tool_messages.append(ToolMessage(
                content=think_tool.invoke(tc["args"]),
                tool_call_id=tc["id"], name="think_tool"))
        elif tc["name"] == "conduct_research":
            print(f"      -> researching: {tc['args']['research_topic'][:70]}")
            result = researcher_graph.invoke({
                "researcher_messages": [
                    SystemMessage(content=RESEARCHER_PROMPT.format(
                        date=today(), research_topic=tc["args"]["research_topic"]))],
                "research_topic": tc["args"]["research_topic"],
                "tool_call_iterations": 0,
                "raw_notes": [],
            }, config)
            new_notes.append(result["compressed_research"])
            # Feed the supervisor a short receipt; full notes go to the report.
            receipt = result["compressed_research"]
            if len(receipt) > 600:
                receipt = receipt[:600] + " ...[full notes saved]"
            tool_messages.append(ToolMessage(
                content=receipt,
                tool_call_id=tc["id"], name="conduct_research"))

    return Command(
        goto="supervisor",
        update={"supervisor_messages": tool_messages, "notes": new_notes},
    )


def build_supervisor_subgraph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("supervisor_tools", supervisor_tools)
    g.add_edge(START, "supervisor")
    return g.compile()


# --- 3. Researcher subgraph ---
def researcher(state: ResearcherState, config) -> Command:
    cfg = Configuration.from_runnable_config(config)
    model = get_model(cfg.research_model).bind_tools(
        get_researcher_tools(research_complete))
    _throttle()
    response = model.invoke(state["researcher_messages"])
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


def researcher_tools(state: ResearcherState, config) -> Command:  # noqa: F811
    cfg = Configuration.from_runnable_config(config)
    last = state["researcher_messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    over_limit = state.get("tool_call_iterations", 0) >= cfg.max_react_tool_calls
    said_done = any(tc["name"] == "research_complete" for tc in tool_calls)
    if not tool_calls or said_done or over_limit:
        return Command(goto="compress_research", update={})

    tool_messages, raw = [], []
    for tc in tool_calls:
        if tc["name"] == "web_search":
            out = web_search.invoke({
                "queries": tc["args"]["queries"],
                "max_results": cfg.max_search_results})
            raw.append(out)
            tool_messages.append(ToolMessage(
                content=out, tool_call_id=tc["id"], name="web_search"))
        elif tc["name"] == "think_tool":
            tool_messages.append(ToolMessage(
                content=think_tool.invoke(tc["args"]),
                tool_call_id=tc["id"], name="think_tool"))

    return Command(
        goto="researcher",
        update={"researcher_messages": tool_messages, "raw_notes": raw},
    )


def compress_research(state: ResearcherState, config) -> dict:
    cfg = Configuration.from_runnable_config(config)
    model = get_model(cfg.compress_model, max_tokens=1024)
    raw = "\n\n".join(state.get("raw_notes", []))
    if len(raw) > 5000:
        raw = raw[:5000] + " ..."
    prompt = COMPRESS_PROMPT.format(
        date=today(), research_topic=state["research_topic"], raw_notes=raw)
    _throttle("      summarizing findings...")
    compressed = model.invoke([HumanMessage(content=prompt)]).content
    return {"compressed_research": compressed}


def build_researcher_subgraph():
    g = StateGraph(ResearcherState)
    g.add_node("researcher", researcher)
    g.add_node("researcher_tools", researcher_tools)
    g.add_node("compress_research", compress_research)
    g.add_edge(START, "researcher")
    g.add_edge("compress_research", END)
    return g.compile()


researcher_graph = build_researcher_subgraph()
supervisor_graph = build_supervisor_subgraph()


# --- 4. Write ---
def final_report_generation(state: AgentState, config) -> dict:
    cfg = Configuration.from_runnable_config(config)
    model = get_model(cfg.final_report_model, max_tokens=4000)
    notes = "\n\n---\n\n".join(state.get("notes", []))
    if len(notes) > 9000:
        notes = notes[:9000] + " ..."
    prompt = FINAL_REPORT_PROMPT.format(
        date=today(), research_brief=state.get("research_brief", ""), notes=notes)
    _throttle("[3/3] WRITE: generating the final report...")
    report = model.invoke([HumanMessage(content=prompt)]).content
    return {"final_report": report, "messages": [AIMessage(content=report)]}


# --- Top-level graph ---
def run_supervisor(state: AgentState, config) -> dict:
    result = supervisor_graph.invoke({
        "supervisor_messages": state["supervisor_messages"],
        "research_brief": state["research_brief"],
        "notes": [],
        "research_iterations": 0,
    }, config)
    return {"notes": result.get("notes", [])}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("write_research_brief", write_research_brief)
    g.add_node("supervisor_subgraph", run_supervisor)
    g.add_node("final_report_generation", final_report_generation)
    g.add_edge(START, "write_research_brief")
    g.add_edge("supervisor_subgraph", "final_report_generation")
    g.add_edge("final_report_generation", END)
    return g.compile()


graph = build_graph()
