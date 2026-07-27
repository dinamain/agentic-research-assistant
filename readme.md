Agentic Research & Automation Assistant# Agentic Research & Automation Assistant

A multi-tool AI agent that reasons about a task, plans an approach, decides which tools to use, acts on that decision, observes the result, and self-corrects — built with LangGraph instead of a fixed, linear pipeline.

**Repo:** https://github.com/dinamain/agentic-research-assistant
**Status:** In active development — core reasoning loop, planning layer, multi-tool routing, and retry handling are built and tested. Not yet deployed.

Built with LangGraph · LangChain · Groq · Tavily · ChromaDB · FastEmbed

---

## What It Does

A RAG pipeline follows the same path every time: retrieve, then generate. Real research tasks don't work that way — sometimes you need to search the web, sometimes check stored documents, sometimes both, in an order that depends on the question itself.

This agent:

1. **Plans** — breaks the task into concrete sub-steps before doing any work
2. **Reasons** — decides whether it needs a tool, and which one, based on the conversation and the plan
3. **Acts** — calls the chosen tool (web search or document search)
4. **Observes** — reads the tool's result and decides what to do next: answer, or call another tool
5. **Recovers** — if a tool call fails or comes back malformed, retries with a corrective instruction, up to a cap, before falling back to an honest error message instead of crashing

---

## Architecture

START
  ↓
Planner node — LLM classifies the input:
  ├─ Conversational (no research/tools needed) → outputs exact sentinel "NO_PLAN_NEEDED"
  └─ Genuine task → outputs a short numbered plan (advisory context for the agent, not enforced step-by-step)
  ↓
Agent node ←──────────────────────────────────────────┐
  │                                                    │
  ├─ If plan == "NO_PLAN_NEEDED":                      │
  │     → plain llm (no tools bound) responds directly │
  │     conversationally → END                         │
  │                                                     │
  └─ Otherwise, llm_with_tools decides:                 │
       - Answer directly if already confident, or if    │
         the answer is already present earlier in the   │
         conversation history (checked BEFORE routing    │
         rules below, even for possessive questions)     │
       - Call search_documents (personal/possessive       │
         questions — "my project", "her resume" — when    │
         the answer isn't already in conversation history) │
       - Call tavily_search (current events, public info)   │
       - Call both in parallel if the plan shows they're      │
         independent                                           │
         │                                                     │
         ├─ No tool needed → END                                │
         └─ Tool call requested                                 │
               ↓                                                │
         Tools node                                             │
           ├─ search_documents (ChromaDB retriever:             │
           │     query rewrite → retrieve k=15 → cross-encoder  │
           │     re-rank → relevance threshold (-3.0) → top 5)  │
           └─ tavily_search (web search: Tavily →               │
                 relevance-score filter (>0.3) → top 3)         │
               ↓                                                │
         back to Agent node ────────────────────────────────────┘

Error handling (wraps the LLM call inside Agent node):
  - RateLimitError / APIConnectionError → capped retry (2), exponential/short backoff, graceful fallback message on exhaustion
  - BadRequestError ("tool_use_failed") → capped retry (2), graceful fallback message on exhaustion
  - Any other exception (including AuthenticationError) → logged loudly (traceback + warning), safe generic fallback message returned — never crashes the user-facing request
  - Planner node has separate, lighter error handling: any failure degrades to "no plan available, proceeding reactively" rather than retrying, since planning is advisory only

Persistence (SqliteSaver, keyed by thread_id):
  - Every node's state is checkpointed to disk (checkpoints.db) after each step
  - A new app.invoke() call with the same thread_id restores full prior state
    (messages accumulate via the add_messages reducer; plan and tool_retry_count
    overwrite, since only their latest value matters)
  - State survives not just repeated calls within one process, but genuine
    process restarts — proven by running two separate Python processes
    against the same thread_id and the same checkpoints.db file
  - thread_id has no meaning to LangGraph beyond being a lookup key — the
    application layer is responsible for deriving it from authenticated,
    server-verified user/session data, never accepting it directly from a client

State (`AgentState`) carries:
  - `messages` (Annotated[list, add_messages] — append-only, full conversation
     and tool-call history; this is what lets the agent recall prior turns,
     both within a run and across checkpointed sessions)
  - `plan` (plain overwrite — set once per turn by the planner, either a
     numbered plan or the "NO_PLAN_NEEDED" sentinel)
  - `tool_retry_count` (plain overwrite — incremented on retryable failure,
     explicitly reset to 0 on every exit path from Agent node, whether success
     or exhausted-retry fallback, so it never carries stale state into the
     next turn)


## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Reasoning / tool-selection LLM | Groq (Llama 3.3-70b-versatile) |
| Query-rewrite LLM (cheaper, high-volume) | Groq (Llama 3.1-8b-instant) |
| Web search | Tavily |
| Document retrieval | ChromaDB + FastEmbed (BAAI/bge-small-en-v1.5) |
| Re-ranking | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Persistence | SQLite (via langgraph-checkpoint-sqlite) — survives process restarts |
---

## Project Structure

agentic-research-assistant/
├── agent.py # State, graph, planner, agent node, retry logic, tools binding
├── retriever_tool.py # search_documents tool: rewrite → retrieve → re-rank → filter
├── ingest.py # PDF ingestion into this project's own ChromaDB (self-contained)
├── run_ingest.py # Script to ingest a document
├── .gitignore
└── README.md

This project keeps its own ChromaDB and ingestion pipeline, deliberately separate from the RAG Document Q&A project — each project is independently runnable and deployable, with no cross-project file-path dependency.

---

## Getting Started

### Prerequisites

- Python 3.12+

### 1. Clone the repo

```bash
git clone https://github.com/dinamain/agentic-research-assistant.git
cd agentic-research-assistant
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install langgraph langchain-groq langchain-tavily langchain-chroma langchain-community fastembed sentence-transformers python-dotenv pypdf
```

### 4. Set up environment variables

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

### 5. Ingest a document (optional — for testing `search_documents`)

```bash
python run_ingest.py
```

### 6. Run the agent

```bash
python agent.py
```

---

## Key Design Decisions

**Why LangGraph instead of a LangChain chain?**
A chain is a fixed pipeline walked the same way every time. An agent's control flow needs loops (reason → act → observe → repeat) and branches decided by the LLM's own output at runtime — not something a static chain or a hardcoded if/else can represent cleanly.

**Why does `AgentState.messages` use an append reducer but `tool_retry_count` doesn't?**
The LLM's next decision depends on the full conversation history, so messages must accumulate. Retry count only needs its current value — its history is irrelevant to any decision — so it overwrites.

**Why start with a mock tool before wiring in real search?**
Isolates whether the graph mechanics work from whether a real API integration works — proving the loop first meant later bugs were clearly attributable to the real tool, not the graph.

**Why calibrate tool-calling with both a docstring and a system prompt?**
The model initially called the search tool even for trivial facts it already knew (e.g. "what color is the sky"). Tool-calling is probabilistic, driven by how a tool's description reads against the conversation — not a rule engine. A tighter docstring plus an explicit system-prompt instruction ("you are not required to use tools") fixed this, verified by testing both a trivial question (no tool call) and a genuine current-events question (correct tool call).

**Why upgrade from an 8B to a 70B model for the agent's reasoning step?**
Adding a second tool caused the 8B model to generate malformed tool-call syntax that the API rejected outright — a known reliability gap for small models on multi-tool selection. The 70B model resolved the malformed-syntax crash, though not every case (see retry handling below) — model capacity should match task complexity, not default to the cheapest option everywhere.

**Why does the system prompt explicitly prioritize document search for possessive/personal language?**
Found a serious failure: the agent used web search for a question about "the SwiftChat project" — a name that collides with unrelated public products. The web results were about a different SwiftChat entirely, and the agent reported them as fact with no hedging, unlike its usual honest "not covered" behavior. Adding an explicit rule — possessive/personal phrasing always routes to document search first — fixed this, verified in both directions (personal question → documents, public/current-events question → web search).

**Why does the agent sometimes call two tools in parallel instead of looping sequentially?**
Once a plan identifies that two independent pieces of information are needed, the model can request both tool calls in a single turn rather than looping twice — LangGraph's `ToolNode` handles this natively. This is faster, but risks a real problem: if one tool's query should depend on the other's result, parallel calling can't express that dependency. Testing confirmed this — a web search fired before a document lookup was too generic to be useful — but also showed the agent's own reasoning loop can self-correct on a second turn once it sees an inadequate result.

**Why cap retries instead of retrying indefinitely?**
Even the 70B model occasionally produces malformed tool-call syntax that crashes the underlying API call. An uncapped or unprotected retry can crash again on the retry itself. `tool_retry_count` in state tracks attempts, retries up to a hard cap, and falls back to an honest error message to the user instead of crashing the whole process — a real production requirement, not just correctness.

**Why filter tool results by relevance score before they reach the LLM?**
Raw, unfiltered search results — including low-relevance noise — were sometimes cited by the LLM as if directly relevant, overstating claims of specificity without fabricating facts outright. Filtering both tools' outputs by an empirically-chosen relevance threshold (cross-encoder score for documents, Tavily's own score for web search) removes weak matches before the LLM ever sees them, rather than relying on the LLM to self-filter.

---

## What I Learned Building This

- **Tool-calling is probabilistic, not rule-based** — a generic tool description makes a model trigger-happy even for facts it already knows; description precision is the actual lever, not a code fix.
- **Model size matters specifically for multi-tool selection** — an 8B model handled single-tool binary decisions fine but failed to even format valid syntax once a second tool was added; a 70B model reduced but did not eliminate this failure class.
- **Confident hallucination is a distinct, more dangerous failure mode than honest uncertainty** — every earlier failure in this project involved the model correctly admitting it didn't know something; the wrong-tool-routing case produced a fluent, specific, entirely fabricated answer with no hedge at all, because of a real-world name collision.
- **Parallel tool calling can silently violate real dependencies between calls** — verified this directly by constructing a question where one tool's query needed the other tool's result first, and watching the first attempt underperform before the agent's own loop corrected it on a second turn.
- **A single retry isn't enough if the retry path itself isn't protected** — an early retry implementation crashed on its own retry attempt, since only the original call was wrapped in error handling.
- **Unfiltered search results get cited as more specific than they are** — even without inventing facts, an LLM can present low-relevance results as directly relevant unless they're filtered out first; this is a distinct problem from hallucination and needs a different fix (relevance thresholds, not better prompting).
- **Planning is advisory, not enforced, in this design** — a planner node gives the reactive loop useful upfront structure, but the graph doesn't force the agent to complete plan steps in order; a stricter architecture would track plan-step completion explicitly in state.

---

## Status & Next Steps

**Done:**
- Core ReAct loop (LangGraph state graph, conditional tool routing)
- Real web search (Tavily) and document retrieval (ChromaDB + re-ranking) as tools
- Planning layer (advisory, pre-loop task decomposition)
- Capped retry handling for malformed tool-call syntax
- Relevance filtering on both tools' outputs

**Not yet built:**
- Broader error handling (network failures, tool-execution exceptions beyond malformed LLM syntax)
- Persistent memory across sessions (LangGraph checkpointing)
- Streaming responses via FastAPI
- Automated evaluation harness
- Deployment

- **Persistence and accumulation are different claims, and need different tests.** Proving state survives a process restart (not just a second invoke() call in the same script) required running two genuinely separate Python processes against the same thread_id — one that wrote state and fully exited, and a second, freshly-started process that could only have recalled the fact from disk. That's the actual distinction between an in-memory checkpointer and a persisted one.
---


## Known Limitations

**No pruning of old conversation threads.** `checkpoints.db` grows unboundedly — nothing currently deletes or archives old threads, so a long-running deployment would need a retention policy (e.g., delete threads inactive for 90+ days) to avoid unbounded disk growth.

**No authentication layer connecting users to thread_id.** This project currently uses hardcoded thread_id strings for testing. In a real multi-user deployment, thread_id must be deterministically derived from a server-verified, authenticated user session — never accepted directly from the client — since anyone who could supply an arbitrary thread_id could read another user's conversation history. This project demonstrates the persistence mechanism; the auth/session layer that would make it production-safe isn't built here.

--

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)