EVAL_CASES = [
    # --- Tool routing correctness ---
    {
        "id": "possessive_routes_to_documents",
        "input": "What tech stack did I use for SwiftChat?",
        "check": "tool_called",
        "expected_tool": "search_documents",
        "must_not_contain": ["Angular", "Vue", "Flask", "MongoDB"],  # guards against fabricated/wrong stack
    },
    {
        "id": "current_events_routes_to_web",
        "input": "What's the latest LangGraph version?",
        "check": "tool_called",
        "expected_tool": "tavily_search",
    },
    {
        "id": "trivial_fact_needs_no_tool",
        "input": "What color is the sky?",
        "check": "no_tool_called",
    },

    # --- History-first precedence (the fix from earlier this session) ---
    {
        "id": "history_answered_no_tool_needed",
        "input": "My name is Dina and I'm building an agentic assistant.",
        "check": "no_tool_called",
    },

    # --- Groundedness / honesty under uncertainty ---
    {
        "id": "irrelevant_search_results_admits_uncertainty",
        "input": "What's the latest news about the Perseverance Mars rover discovering ancient alien cities?",
        "check": "tool_called",
        "expected_tool": "tavily_search",
        "must_not_contain": ["ancient alien cities", "discovered aliens"],  # must not hallucinate a fake premise as fact
    },

    # --- Content correctness (new check type — see runner update below) ---
    {
        "id": "swiftchat_stack_is_correct",
        "input": "What tech stack did I use for SwiftChat?",
        "check": "answer_contains",
        "expected_terms": ["Django Channels", "Redis"],
    },
    {
        "id": "langgraph_version_is_current",
        "input": "What's the latest LangGraph version?",
        "check": "answer_contains",
        "expected_terms": ["1.2"],  # loosely matches 1.2.x without over-pinning to an exact patch version
    },

    # --- Relevance filtering (the security-vulnerability finding) ---
    {
        "id": "low_relevance_results_rejected",
        "input": "What specific CVEs affect Django Channels and Redis when used together?",
        "check": "tool_called",
        "expected_tool": "tavily_search",
        "must_not_contain": ["CVE-"],  # if no real CVE was found, answer must not fabricate a CVE number
    },
    {
    "id": "name_collision_person_no_hallucination",
    "input": "Tell me about Dina Usman",
    "check": "tool_called",
    "expected_tool": "search_documents",
    "must_not_contain": ["MMA", "fighter", "Black Cat", "record"],
},
{
    "id": "name_collision_no_possessive_language",
    "input": "Who is Dina Usman?",
    "check": "tool_called",
    "expected_tool": "search_documents",
},
]