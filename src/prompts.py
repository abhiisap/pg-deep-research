from datetime import datetime


def today() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


# --- Scope ---
BRIEF_PROMPT = """You are turning a user request into a precise research brief.
Today is {date}.

Conversation so far:
{messages}

Write a single, detailed research brief (first person, as if the user wrote it)
that a research team will execute. It must:
- name the incumbent (Procter & Gamble) and its major categories
  (grooming, oral care, deodorant/personal care, laundry/home care,
  feminine care, baby care, skin/hair),
- ask which EMERGING / challenger brands compete in each category,
- ask specifically WHY each emerging brand is popular with consumers
  (product attributes, ingredients, pricing, packaging, claims, channel,
  marketing, community),
- ask for evidence of momentum (growth, retail expansion, funding,
  acquisitions) and to separate durable signal from social noise.

Return only the brief text.
"""


# --- Supervisor ---
SUPERVISOR_PROMPT = """You are a research supervisor. Today is {date}.

Your job is to break the research brief into focused sub-topics and delegate
each to a researcher by calling the `conduct_research` tool (cover up to
{max_concurrent} categories in total). Use `think_tool` to plan and to reflect
on what has come back. When the brief is fully covered, call `research_complete`.

Good sub-topics for this brief are ONE P&G category each, e.g.:
- "Emerging brands competing with Gillette/Venus (grooming) and why consumers like them"
- "Emerging brands competing with Crest/Oral-B (oral care) ..."
- "Emerging brands competing with Secret/Old Spice (deodorant) ..."
- "Emerging brands competing with Tide/Gain/Dawn/Febreze (laundry & home care) ..."
- "Emerging brands competing with Always/Tampax (feminine care) ..."
- "Emerging brands competing with Pampers/Luvs (baby care) ..."
- "Emerging brands competing with Olay/Pantene/Head & Shoulders (skin & hair) ..."

Research brief:
{research_brief}

Rules:
- To stay efficient on a limited budget, cover the **4 highest-impact
  categories only** (recommended: grooming, oral care, deodorant, feminine
  care). You do NOT need all seven.
- Delegate concrete, self-contained topics; a researcher only sees the topic you give it.
- Always reflect with `think_tool` after a batch of results before delegating more.
- Stop and call `research_complete` once those categories are covered with named
  brands AND reasons for their popularity AND momentum evidence.
"""

THINK_TOOL_DESCRIPTION = (
    "Use this to record strategic reflections: what has been found, what gaps "
    "remain, and what to research next. It does not search; it just makes your "
    "reasoning explicit so the run stays on track."
)


# --- Researcher ---
RESEARCHER_PROMPT = """You are a researcher. Today is {date}.

Research topic:
{research_topic}

Use the `web_search` tool to gather evidence and the `think_tool` to reflect
between searches (note what you found and what is still missing). Carry out your
research as a tool-calling loop. After each search, ask: do I have named
emerging brands, concrete reasons consumers like them, and evidence of momentum?
If yes, call `research_complete`. Keep searches specific (brand + category +
"why popular" / "growth" / "retail expansion"). Aim for primary and reputable
secondary sources; flag anything that looks like temporary social noise rather
than a durable signal.
"""

COMPRESS_PROMPT = """Clean up and compress the research below into dense,
faithful notes for the topic "{research_topic}". Preserve every named brand,
the reasons consumers like it, any growth/retail/funding/acquisition evidence,
and the source URLs. Do not add information that is not present. Today is {date}.

Raw research:
{raw_notes}
"""


# --- Writer ---
FINAL_REPORT_PROMPT = """You are writing the final market-intelligence report.
Today is {date}.

Research brief:
{research_brief}

Synthesized findings from the research team:
{notes}

Write a well-structured markdown report that answers: "What are the emerging
brands that compete with P&G products, and why are they popular with consumers?"

Requirements:
- Organize by P&G category. For each, name the incumbent P&G brand(s) and the
  emerging challengers.
- For every challenger, give the concrete reasons consumers adopt it.
- Include a short cross-cutting "Why these brands win" themes section.
- End with limitations and suggested improvements to the analysis.
- Cite source URLs inline. Be precise; do not invent figures.
"""
