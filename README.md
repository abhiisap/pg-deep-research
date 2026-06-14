# P&G Emerging-Competitor Deep Research Agent

A small multi-agent research agent built with LangGraph that answers one question:

> What are the emerging brands that compete with P&G products, and why are they popular with consumers?

It follows the Scope → Research → Write pattern from LangChain's
[Open Deep Research](https://www.langchain.com/blog/open-deep-research) and the
[LangChain Academy "Deep Research with LangGraph"](https://academy.langchain.com/courses/deep-research-with-langgraph)
course. A scope step turns the question into a research brief, a supervisor splits
that brief into one sub-topic per P&G category and hands each to a researcher
sub-agent, and a writer pulls the notes together into a sourced report.

## What it found

I ran it on the free model tier (Llama 3.3 on Groq) with a deliberately small
search budget, so the findings are a starting point rather than an exhaustive
scan. The full report is in [`results/RESULTS.md`](results/RESULTS.md).

- Grooming (vs Gillette/Venus): Billie and Schick came up as challengers.
- Deodorant (vs Secret/Old Spice): Native, plus smaller natural-deodorant names
  like Routine, No Pong, Green Beaver and Myro. The recurring reason consumers
  switch is aluminium-free, natural-ingredient positioning.
- Oral care and feminine care: the run surfaced market-size signals but no
  named challengers, mostly because the search budget was too shallow to dig in.

The common thread across what it did find: cleaner/"better-for-you" ingredients
plus direct-to-consumer convenience and social-media-led brand building.

For the full write-up of the approach and findings, see [`approach.md`](approach.md).

## How it works

```
question ──▶ scope ──▶ supervisor ──▶ researcher (one per category) ──▶ write ──▶ results/RESULTS.md
```

- Scope writes a research brief from the question.
- Supervisor breaks the brief into category sub-topics and delegates each to a
  researcher. It uses a `think_tool` to plan between delegations, and its loop is
  capped so the run terminates.
- Each researcher runs its own search/reflect loop (`web_search` + `think_tool`),
  then compresses what it found into source-cited notes. Giving each researcher
  its own context window is what lets the run cover several categories without one
  giant prompt.
- Write synthesizes all the notes into the final report.

In this implementation the supervisor delegates categories one at a time rather
than truly in parallel, which keeps the run within the free-tier rate limits.

| File | Role |
|------|------|
| `src/deep_researcher.py` | Graph wiring: scope, supervisor, researcher, writer |
| `src/state.py` | State definitions and the override reducer |
| `src/prompts.py` | All prompts |
| `src/utils.py` | `web_search` (Tavily), `think_tool`, model setup |
| `src/configuration.py` | Models, search API, iteration limits |
| `run.py` | Entry point that runs the agent and saves the report |
| `langgraph.json` | Lets you open the graph in LangGraph Studio |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add GROQ_API_KEY and TAVILY_API_KEY (both free)
python run.py               # runs the agent, prints and writes results/RESULTS_live.md
```

On the free tier the run pauses between steps to stay under the per-minute token
limit, so it takes a few minutes. Progress prints as it goes.

To explore the graph visually:

```bash
pip install "langgraph-cli[inmem]"
langgraph dev
```

Models, search backend, and loop limits are set in `src/configuration.py` and can
be overridden with the env vars listed in `.env.example`. To use a different
provider, change the model strings; to use a different search tool, edit
`web_search` in `src/utils.py`.
