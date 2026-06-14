"""State definitions for the deep research agent."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


def override_reducer(current, new):
    """Append by default, or replace when given {"type": "override", "value": ...}."""
    if isinstance(new, dict) and new.get("type") == "override":
        return new["value"]
    return operator.add(current, new)


class AgentState(MessagesState):
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    supervisor_messages: Annotated[list, override_reducer]
    final_report: str


class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list, override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int


class ResearcherState(TypedDict):
    researcher_messages: Annotated[list, operator.add]
    research_topic: str
    tool_call_iterations: int
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]
