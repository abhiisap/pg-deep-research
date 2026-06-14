"""Runtime configuration, overridable via env vars or the invoke config."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Configuration:
    # Models default to Groq's free tier (set GROQ_API_KEY in .env).
    scope_model: str = os.getenv("SCOPE_MODEL", "groq:llama-3.3-70b-versatile")
    research_model: str = os.getenv("RESEARCH_MODEL", "groq:llama-3.3-70b-versatile")
    compress_model: str = os.getenv("COMPRESS_MODEL", "groq:llama-3.3-70b-versatile")
    final_report_model: str = os.getenv("FINAL_MODEL", "groq:llama-3.3-70b-versatile")

    search_api: str = os.getenv("SEARCH_API", "tavily")
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "2"))

    # Kept small to stay under the free-tier per-minute token budget.
    max_concurrent_research_units: int = int(os.getenv("MAX_CONCURRENT", "4"))
    max_researcher_iterations: int = int(os.getenv("MAX_SUPERVISOR_ITERS", "4"))
    max_react_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", "2"))

    @classmethod
    def from_runnable_config(cls, config: dict | None = None) -> "Configuration":
        config = config or {}
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        kwargs = {f: configurable[f] for f in cls.__dataclass_fields__ if f in configurable}
        return cls(**kwargs)
