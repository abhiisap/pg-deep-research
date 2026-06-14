"""Tools and helpers: web search, the reflection tool, and model construction."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from .configuration import Configuration
from .prompts import THINK_TOOL_DESCRIPTION


def get_model(model_name: str, max_tokens: int = 8000):
    # Providers name the output-length param differently.
    if model_name.startswith("google_genai:"):
        return init_chat_model(model=model_name, max_output_tokens=max_tokens)
    if model_name.startswith("groq:"):
        return init_chat_model(model=model_name, max_tokens=max_tokens, max_retries=5)
    return init_chat_model(model=model_name, max_tokens=max_tokens)


@tool(description=THINK_TOOL_DESCRIPTION)
def think_tool(reflection: str) -> str:
    """Record a reflection during research (no external action)."""
    return f"Reflection recorded:\n{reflection}"


@tool
def web_search(queries: list[str], max_results: int = 5) -> str:
    """Search the web (Tavily) and return formatted results."""
    from langchain_tavily import TavilySearch

    search = TavilySearch(max_results=max_results, topic="general")
    blocks: list[str] = []
    for q in queries:
        res = search.invoke({"query": q})
        for r in res.get("results", []):
            content = (r.get("content", "") or "").strip()
            if len(content) > 700:
                content = content[:700] + " ..."
            blocks.append(
                f"### {r.get('title', 'Untitled')}\n"
                f"URL: {r.get('url', '')}\n"
                f"{content}\n"
            )
    return "\n".join(blocks) if blocks else "No results found."


def get_supervisor_tools(conduct_research, research_complete):
    return [conduct_research, think_tool, research_complete]


def get_researcher_tools(research_complete):
    return [web_search, think_tool, research_complete]
