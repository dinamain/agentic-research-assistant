# Agentic Research & Automation Assistant

A multi-tool AI agent that reasons about a task, plans an approach, decides which tools to use, acts on that decision, observes the result, and self-corrects — built with LangGraph instead of a fixed, linear pipeline.

**Repo:** https://github.com/dinamain/agentic-research-assistant
**Live demo:** https://agentic-assistant-frontend.vercel.app
**Status:** Deployed and live. Core reasoning loop, planning layer, multi-tool routing, retry handling, persistence, streaming, document upload, and an automated eval harness are all built and tested end-to-end — including in the live deployed environment, not just locally.

Built with LangGraph · LangChain · Groq · Tavily · ChromaDB · FastEmbed · FastAPI · React

---

## What It Does

A RAG pipeline follows the same path every time: retrieve, then generate. Real research tasks don't work that way — sometimes you need to search the web, sometimes check stored documents, sometimes both, in an order that depends on the question itself.

This agent:

1. **Plans** — breaks the task into concrete sub-steps before doing any work
2. **Reasons** — decides whether it needs a tool, and which one, based on the conversation and the plan
3. **Acts** — calls the chosen tool (web search or document search)
4. **Observes** — reads the tool's result and decides what to do next: answer, or call another tool
5. **Recovers** — if a tool call fails or comes back malformed, either repairs it directly (see below) or retries with a corrective instruction up to a cap, before falling back to an honest error message instead of crashing

A user can also **upload their own PDF** through the live demo, which gets ingested into the agent's document store and becomes searchable via the `search_documents` tool.

---

## Architecture

```
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
       - Call search_documents (personal/possessive,      │
         or anything that is or could be Dina's own —      │
         her name, her projects, her background — even      │
         without explicit possessive language)               │
       - Call tavily_search (current events, public info)   │
       - Call both in parallel if the plan shows they're      │
         independent                                           │
         │                                                     │
         ├─ No tool needed → END                                │
         └─ Tool call requested                                 │
               ↓                                                │
         Tools node                                             │
           ├─ search_documents (ChromaDB retriever: query        │
           │     rewrite → retrieve k=5 → vector-similarity       │
           │     distance filter, threshold 0.87)                 │
           └─ tavily_search (web search: Tavily →               │
                 relevance-score filter (>0.3) → top 3)         │
               ↓                                                │
         back to Agent node ────────────────────────────────────┘

POST /upload → PDF → PyPDF extract → clean_text → chunk → contextual
               header → FastEmbed embed → same shared ChromaDB used
               by search_documents (no duplicate model instance)
```

**Error handling** (wraps the LLM call inside Agent node):
- `RateLimitError` / `APIConnectionError` → capped retry (2), exponential/short backoff, graceful fallback message on exhaustion
- `BadRequestError` / `APIError` ("tool_use_failed" / "Failed to call a function") → **first**, attempt direct recovery (see below); if that fails, capped retry (2), graceful fallback message on exhaustion. Both exception types are caught because the same underlying malformed-tool-call failure surfaces differently depending on whether the graph is invoked normally or streamed
- Any other exception (including `AuthenticationError`) → logged loudly (traceback + warning), safe generic fallback message returned — never crashes the user-facing request
- Planner node has separate, lighter error handling: any failure degrades to "no plan available, proceeding reactively" rather than retrying, since planning is advisory only

**Malformed tool-call recovery:** rather than only retrying on a malformed tool call, the agent inspects Groq's `failed_generation` field and attempts to directly parse a usable tool call out of it. This was built after diagnosing a specific, reliably-reproducing failure — see Key Design Decisions.

**Persistence** (SqliteSaver, keyed by `thread_id`):
- Every node's state is checkpointed to disk (`checkpoints.db`) after each step
- A new `app.invoke()` call with the same `thread_id` restores full prior state (`messages` accumulate via the `add_messages` reducer; `plan` and `tool_retry_count` overwrite, since only their latest value matters)
- State survives not just repeated calls within one process, but genuine process restarts — proven by running two separate Python processes against the same `thread_id` and the same `checkpoints.db` file
- `thread_id` has no meaning to LangGraph beyond being a lookup key — the application layer is responsible for deriving it from authenticated, server-verified user/session data, never accepting it directly from a client

**Streaming** (FastAPI + Server-Sent Events):
- `stream_mode="updates"` surfaces node-level progress (planner → agent → tools → agent) — useful for showing "Searching your documents..." style progress in a UI
- `stream_mode="messages"` surfaces token-level streaming of the final answer — filtered to the `agent` node's chunks with non-empty `content`, since a single turn involves multiple separate LLM calls (planning, tool-call generation, query rewriting) and only one of them is the user-facing answer
- Tested end-to-end over a real HTTP connection, both locally and against the live Render deployment, via a `/chat/stream` endpoint

**State** (`AgentState`) carries:
- `messages` (`Annotated[list, add_messages]` — append-only, full conversation and tool-call history; this is what lets the agent recall prior turns, both within a run and across checkpointed sessions)
- `plan` (plain overwrite — set once per turn by the planner, either a numbered plan or the `"NO_PLAN_NEEDED"` sentinel)
- `tool_retry_count` (plain overwrite — incremented on retryable failure, explicitly reset to 0 on every exit path from Agent node, whether success or exhausted-retry fallback, so it never carries stale state into the next turn)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Reasoning / tool-selection LLM | Groq (Llama 3.3-70b-versatile, temperature=0) |
| Query-rewrite LLM (cheaper, high-volume) | Groq (Llama 3.1-8b-instant) |
| Web search | Tavily |
| Document retrieval | ChromaDB + FastEmbed (BAAI/bge-small-en-v1.5, ONNX) |
| Relevance filtering | Vector-similarity distance threshold (calibrated empirically) |
| Persistence | SQLite (via `langgraph-checkpoint-sqlite`) — survives process restarts |
| API / Streaming | FastAPI + Server-Sent Events |
| Frontend | React (Vite) |
| Deployment | Render (API) · Vercel (frontend) |

---

## Project Structure

```
agentic-research-assistant/
├── agent.py            # State, graph, planner, agent node, retry + recovery logic, tools binding
├── retriever_tool.py    # search_documents tool: rewrite → retrieve → relevance filter
├── main.py              # FastAPI app: /chat/stream (streaming), /upload
├── ingest.py            # PDF ingestion into this project's own ChromaDB (self-contained)
├── run_ingest.py         # Script to ingest a document locally
├── eval_cases.py         # Eval harness test cases
├── eval_runner.py        # Eval harness runner
├── .gitignore
└── README.md
```

This project keeps its own ChromaDB and ingestion pipeline, deliberately separate from the RAG Document Q&A project — each project is independently runnable and deployable, with no cross-project file-path dependency.

---

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js (for the frontend)

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
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Ingest a document (optional — for testing `search_documents`)
```bash
python run_ingest.py
```

### 6. Run the agent directly
```bash
python agent.py
```

### 7. Run the API server (streaming + upload)
```bash
uvicorn main:app --reload
```

### 8. Run the eval suite
```bash
python eval_runner.py
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
Adding a second tool caused the 8B model to generate malformed tool-call syntax that the API rejected outright — a known reliability gap for small models on multi-tool selection. The 70B model reduced this significantly, though not to zero (see the recovery mechanism below) — model capacity should match task complexity, not default to the cheapest option everywhere.

**Why force `temperature=0` on the main reasoning model?**
Groq's own documentation notes that lower temperature reduces the chance of malformed tool calls by making generation more deterministic. This was applied after diagnosing the malformed-tool-call issue below — it reduces the failure rate but does not eliminate it, since the underlying cause turned out to be a formatting quirk, not pure sampling randomness.

**Why does the system prompt explicitly prioritize document search for possessive/personal language — and later, for *any* name that could collide?**
Found a serious failure: the agent used web search for a question about "the SwiftChat project" — a name that collides with unrelated public products — and reported the wrong results as fact with no hedging. Adding a rule that possessive/personal phrasing always routes to document search first fixed this. Later, a broader case surfaced: "Tell me about Dina Usman" (no possessive language at all) triggered a web search that found a real, different person with a similar name, and the agent's answer blended both people's information — a fabricated biography attributed to Dina, including invented professional details. The routing rule was generalized from "possessive language" to "any named entity that is or could be Dina's own," and further tightened so that once `search_documents` gives a sufficient answer, the agent doesn't also surface an unrelated same-named person's details — even with a disclaimer distinguishing them. Verified end-to-end: the same question that previously produced a fabricated, blended biography now returns only accurate, resume-sourced content.

**Why does the agent sometimes call two tools in parallel instead of looping sequentially?**
Once a plan identifies that two independent pieces of information are needed, the model can request both tool calls in a single turn rather than looping twice — LangGraph's `ToolNode` handles this natively. This is faster, but risks a real problem: if one tool's query should depend on the other's result, parallel calling can't express that dependency. Testing confirmed this — a web search fired before a document lookup was too generic to be useful — but also showed the agent's own reasoning loop can self-correct on a second turn once it sees an inadequate result.

**Why cap retries instead of retrying indefinitely?**
Even the 70B model occasionally produces malformed tool-call syntax that crashes the underlying API call. An uncapped or unprotected retry can crash again on the retry itself. `tool_retry_count` in state tracks attempts, retries up to a hard cap, and falls back to an honest error message to the user instead of crashing the whole process — a real production requirement, not just correctness.

**Why recover from malformed tool calls directly, instead of only retrying?**
"Tell me about Dina Usman" reliably failed with a malformed tool call across many attempts, even after `temperature=0` and the broadened routing rule — a much higher failure rate than any other tested input, suggesting something specific rather than random flakiness. Inspecting Groq's `failed_generation` error field (rather than just the error type) revealed the actual cause: the model was correctly choosing `search_documents` with the correct arguments, but wrapping them in `<function=search_documents {"query": "Dina Usman"}</function>` — an XML-style tag wrapper — instead of the plain JSON Groq's API strictly requires. This is a documented quirk of `llama-3.3-70b-versatile` on Groq's platform, not a reasoning error. Since the underlying decision was already correct, a targeted fix made more sense than more retries: a recovery step parses the tool name and arguments directly out of the malformed wrapper and constructs a valid tool call from them, turning a fully-exhausted, user-facing failure into a silent, successful recovery.

**Why filter tool results by relevance score before they reach the LLM?**
Raw, unfiltered search results — including low-relevance noise — were sometimes cited by the LLM as if directly relevant, overstating claims of specificity without fabricating facts outright. Filtering both tools' outputs by an empirically-chosen relevance threshold removes weak matches before the LLM ever sees them, rather than relying on the LLM to self-filter.

**Why does `search_documents` use a vector-similarity threshold instead of cross-encoder re-ranking?**
The original implementation used a `sentence-transformers` cross-encoder for re-ranking — but this pulls in a full PyTorch install alongside FastEmbed's ONNX embeddings, and running two heavy model runtimes together exceeded Render's free-tier 512MB memory limit during deployment. The fix: drop the cross-encoder, use Chroma's own similarity-with-score directly, with a threshold calibrated by inspecting real score distributions (0.87, chosen from a clear gap between same-document and different-document scores in testing). This is a deliberate precision-for-memory tradeoff — see Known Limitations.

**Why does streaming's error handling catch a different exception type than normal invocation?**
The same malformed-tool-call failure raises `BadRequestError` when the graph is invoked normally, but `APIError` when streamed — a quirk of the underlying Groq client's streaming code path. Discovered when the retry logic silently stopped firing under streaming; fixed by broadening the catch to both exception types and both their characteristic error-message substrings.

**Why does `/upload` reuse the same vectorstore instance instead of creating a new one?**
The initial implementation created a fresh `Chroma`/`FastEmbedEmbeddings` instance inside the upload handler whenever none was passed in — which meant every upload briefly held a second, duplicate copy of the embedding model in memory alongside the one already loaded by `search_documents`. On Render's free tier, already running close to its 512MB limit, this duplicate load was enough to trigger an out-of-memory crash — independent of how large the uploaded document was; even a two-page resume triggered it. The fix: expose the already-loaded vectorstore from `retriever_tool.py` and have the upload endpoint reuse it directly, eliminating the duplicate model load entirely.

**Why write the uploaded file to a `tempfile.NamedTemporaryFile` instead of a path built from the filename?**
An earlier version wrote the uploaded file to `./{filename}` — during testing, uploading a file that happened to share a name with a file already in the working directory caused the upload's write operation to overwrite that file while it was still being read, corrupting it mid-request. `NamedTemporaryFile` guarantees a unique path every time, structurally eliminating this class of collision rather than relying on users never picking a colliding filename.

---

## What I Learned Building This

- **Tool-calling is probabilistic, not rule-based** — a generic tool description makes a model trigger-happy even for facts it already knows; description precision is the actual lever, not a code fix.
- **Model size matters specifically for multi-tool selection** — an 8B model handled single-tool binary decisions fine but failed to even format valid syntax once a second tool was added; a 70B model reduced but did not eliminate this failure class.
- **Confident hallucination is a distinct, more dangerous failure mode than honest uncertainty** — the most serious failure in this project wasn't the model saying "I don't know," it was a fluent, specific, partially-fabricated biography, produced because a real, different person shares a similar name. Fixing this required generalizing a narrow "possessive language" rule into a broader "any name that could collide" principle.
- **A "malformed tool call" isn't one failure mode — it's whatever the model's output doesn't parse as, and the fix depends on which.** Inspecting the actual `failed_generation` field, rather than just the error type, revealed the model was consistently right about *what* to call and wrong only about *how* to format it — an XML-tag wrapper around otherwise-valid JSON. That distinction meant the real fix was targeted recovery, not just more retries.
- **A high, input-specific failure rate is a signal, not just bad luck.** One question failing far more often than every other tested input was the clue that something specific — not general flakiness — was going on, and was worth digging into with the actual error payload rather than assumed away.
- **Parallel tool calling can silently violate real dependencies between calls** — verified this directly by constructing a question where one tool's query needed the other tool's result first, and watching the first attempt underperform before the agent's own loop corrected it on a second turn.
- **A single retry isn't enough if the retry path itself isn't protected** — an early retry implementation crashed on its own retry attempt, since only the original call was wrapped in error handling.
- **Unfiltered search results get cited as more specific than they are** — even without inventing facts, an LLM can present low-relevance results as directly relevant unless they're filtered out first; this is a distinct problem from hallucination and needs a different fix (relevance thresholds, not better prompting).
- **Planning is advisory, not enforced, in this design** — a planner node gives the reactive loop useful upfront structure, but the graph doesn't force the agent to complete plan steps in order; a stricter architecture would track plan-step completion explicitly in state.
- **Persistence and accumulation are different claims, and need different tests.** Proving state survives a process restart (not just a second `invoke()` call in the same script) required running two genuinely separate Python processes against the same `thread_id` — one that wrote state and fully exited, and a second, freshly-started process that could only have recalled the fact from disk.
- **Streaming exercises a different code path than normal invocation, and errors can behave differently as a result.** The same malformed-tool-call failure raised a different exception type (`APIError` vs. `BadRequestError`) depending on whether the graph was invoked normally or streamed — silently bypassing the existing retry logic until the catch was broadened to handle both.
- **Every added pipeline stage has a resource cost somewhere, not just a speed one.** Two separate features — the cross-encoder reranker, and later the upload endpoint's accidental duplicate embedding model — each independently pushed memory past Render's free-tier limit, for different reasons. Diagnosing the second one required recognizing that document size wasn't the actual variable (a two-page resume failed identically to a full book), which pointed at a structural memory issue rather than a content-size one.
- **A file's destination path matters as much as its content.** A temp-file write path that happened to collide with an existing file's name caused a silent read/write race that corrupted an upload — fixed by guaranteeing a unique path structurally (`tempfile.NamedTemporaryFile`) rather than hoping for no collisions.

---

## Status & Next Steps

**Done:**
- Core ReAct loop (LangGraph state graph, conditional tool routing)
- Real web search (Tavily) and document retrieval (ChromaDB) as tools
- Planning layer (advisory, pre-loop task decomposition)
- Capped retry handling (rate limits, connection errors, malformed tool-call syntax, catch-all)
- Direct recovery from a specific, diagnosed malformed-tool-call pattern (XML-wrapped JSON)
- Relevance filtering on both tools' outputs
- Name-collision routing, generalized beyond explicit possessive language and tightened to exclude unrelated same-named entities entirely
- Persistent memory across sessions (SqliteSaver, proven to survive process restarts)
- Streaming responses via FastAPI (node-level and token-level, tested over real HTTP, locally and live)
- Document upload endpoint, with a shared vectorstore instance and collision-safe temp file handling
- Automated evaluation harness (10 cases covering tool routing, groundedness, name-collision handling, and content correctness, with systemic-failure detection)
- Deployed live: backend on Render, frontend on Vercel

**Not yet built:**
- Per-document scoping for `search_documents` (see Known Limitations)
- Broader error handling for tool-execution exceptions beyond malformed LLM syntax
- Conversation thread pruning / retention policy
- Node-level progress events (e.g. "Searching your documents...") surfaced to the frontend — currently only the final answer streams to the UI, though the underlying node-level stream mode is implemented and tested

---

## Known Limitations

**No pruning of old conversation threads.** `checkpoints.db` grows unboundedly — nothing currently deletes or archives old threads, so a long-running deployment would need a retention policy (e.g., delete threads inactive for 90+ days) to avoid unbounded disk growth.

**No authentication layer connecting users to `thread_id`.** This project currently derives `thread_id` client-side for demo purposes. In a real multi-user deployment, `thread_id` must be deterministically derived from a server-verified, authenticated user session — never accepted directly from the client — since anyone who could supply an arbitrary `thread_id` could read another user's conversation history. This project demonstrates the persistence mechanism; the auth/session layer that would make it production-safe isn't built here.

**No cross-encoder re-ranking on `search_documents` (memory constraint).** This tool uses vector similarity with a distance threshold (< 0.87, empirically calibrated by inspecting the real score distribution between relevant and irrelevant documents) instead of cross-encoder re-ranking, to fit Render's free-tier 512MB memory limit. In practice, this means retrieved chunks are correctly filtered to the right document, but not always precisely ranked by relevance within it.

**`search_documents` has no per-document scoping.** Unlike the RAG Document Q&A project, which filters retrieval by filename, this agent's document search runs across the entire ChromaDB collection — every document ever ingested, mixed together. Uploading multiple documents means a question can't currently be scoped to just one of them; retrieval draws from whatever's been ingested overall.

**Ephemeral storage on Render's free tier.** Both `chroma_db` (uploaded documents) and `checkpoints.db` (conversation history) live on local disk, which does not persist across service restarts or redeploys — the same constraint documented in the RAG Document Q&A project. An uploaded document may not still be available after the service has been idle and restarts.

---

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)
