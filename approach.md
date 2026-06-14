# Approach & Findings (summary paragraph)

**Approach.** I built a multi-agent deep-research agent in LangGraph following
the Scope → Research → Write pattern from LangChain's Open Deep Research (the
LangChain Academy "Deep Research with LangGraph" course). A **Scope** node turns
the question — *"what are the emerging brands that compete with P&G products and
why are they popular with consumers?"* — into a structured research brief. A
**Supervisor** agent then breaks that brief into one sub-topic per P&G category
(grooming, oral care, deodorant, feminine care) and delegates each to a
**Researcher** sub-agent that runs its own tool-calling loop: it issues
`web_search` queries, uses a `think_tool` to reflect on gaps between searches,
then `compress_research` distills its findings into source-cited notes.
Isolating each sub-topic in its own context window and capping the supervisor
and researcher loops keeps the long-horizon run coherent and within the
rate limits of the free model tier (Llama 3.3 on Groq) I used. A final **Write**
node synthesizes all the notes into one category-by-category report.

**What I found.** The agent identified emerging challengers to P&G across
several categories. In grooming it surfaced **Billie** as a challenger to
Gillette/Venus; in deodorant it surfaced **Native** (notably, already acquired
by P&G) competing with Secret/Old Spice, alongside smaller natural-deodorant
players; and it pulled in market-size and share signals for oral care and
feminine care. The recurring reason consumers adopt these brands is a
clean/better-for-you positioning — especially **aluminum-free and natural
ingredients** — combined with direct-to-consumer convenience and social-media-led
brand building. The agent also flagged its own gaps: with only a shallow,
free-tier search budget it under-covered some categories and occasionally pulled
weaker sources, which points directly to the improvements I'd make next (deeper
search, access to community signals like Reddit/TikTok, and a momentum score to
separate durable trends from short-lived buzz). The full agent output is in
[`results/RESULTS.md`](results/RESULTS.md).
