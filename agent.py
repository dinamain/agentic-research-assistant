from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_retry_count: int
    plan: str


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from retriever_tool import search_documents

load_dotenv()

_raw_tavily = TavilySearch(max_results=5)


@tool
def tavily_search(query: str) -> str:
    """Search the web for current events, news, or general public information.
    Use this for anything requiring up-to-date information — NOT for Dina's personal
    documents or projects (use search_documents for those instead)."""

    try:
        raw_result = _raw_tavily.invoke({"query": query})
        results = raw_result.get("results", []) if isinstance(raw_result, dict) else []

        relevant = [r for r in results if r.get("score", 0) > 0.3]

        if not relevant:
            return "No sufficiently relevant web results found for this query."

        formatted = "\n\n---\n\n".join(
            f"{r['title']}\n{r['url']}\n{r['content'][:500]}"
            for r in relevant[:3]
        )
        return formatted

    except Exception as e:
        print(f"⚠️ tavily_search failed internally: {e}")
        return f"Web search encountered an error and could not complete: {e}"


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools([tavily_search, search_documents])


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful research assistant. You have access to two tools: "
    "'tavily_search' for current events, news, or general public web information, and "
    "'search_documents' for anything about Dina's own projects, resume, or personal documents.\n\n"
    "IMPORTANT: If the answer to the question is already present earlier in this conversation "
    "(something the user already told you, or something you already found and stated in a "
    "previous turn), answer directly from that conversation history — do NOT call any tool, "
    "even if the question uses possessive language like 'my' or 'this'. Tools are for finding "
    "NEW information you don't already have, not for re-confirming something already said.\n\n"
    "If the question refers to something personal or possessive — 'my project', "
    "'the SwiftChat project', 'her resume', 'this document' — and the answer is NOT already "
    "in the conversation history, ALWAYS use 'search_documents' FIRST, even if the name might "
    "also match something public on the web. Do not use 'tavily_search' for anything that "
    "sounds like it belongs to Dina personally, since public results with a similar or "
    "identical name could be about a completely different, unrelated project — reporting "
    "those as fact would be a serious error.\n\n"
    "You are not required to use any tool — answer directly from your own knowledge whenever "
    "you're confident and the question doesn't require current information or document lookup. "
    "When you do search, formulate specific, keyword-focused queries rather than "
    "echoing the user's full question."
))


def planner_node(state: AgentState):
    user_question = state["messages"][-1].content

    planning_prompt = f"""Determine if the following input requires research, tool use, or a 
multi-step plan to respond to, OR if it's conversational input (a statement, greeting, 
personal introduction, opinion, or anything not actually asking for information or action).

If it's conversational and does NOT require research or tools, respond with EXACTLY:
NO_PLAN_NEEDED

If it genuinely requires research or action, break it into a short numbered list of concrete 
steps. Do not invent unnecessary steps for simple questions.

Input: {user_question}

Response:"""

    plan_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

    try:
        result = plan_llm.invoke(planning_prompt)
        return {"plan": result.content}
    except Exception as e:
        print(f"⚠️ Planning step failed: {e}. Proceeding without a plan.")
        return {"plan": "No plan available due to an error — proceeding reactively."}

from groq import BadRequestError, RateLimitError, APIConnectionError
import time

def agent_node(state: AgentState):
    plan = state.get("plan", "No plan yet.")

    if plan.strip() == "NO_PLAN_NEEDED":
        base_messages = [
            SystemMessage(content="You are a helpful, friendly assistant. Respond naturally and conversationally — no tools are needed for this input.")
        ] + state["messages"]
        response = llm.invoke(base_messages)  # note: plain llm, NOT llm_with_tools
        return {"messages": [response], "tool_retry_count": 0}

    plan_context = SystemMessage(content=f"Here is the plan for this task:\n{plan}")
    base_messages = [SYSTEM_PROMPT, plan_context] + state["messages"]

    retry_count = state.get("tool_retry_count", 0)
    max_retries = 2

    try:
        response = llm_with_tools.invoke(base_messages)
        return {"messages": [response], "tool_retry_count": 0}

    except RateLimitError as e:
        if retry_count >= max_retries:
            fallback = AIMessage(content="I'm currently rate-limited and unable to complete this request. Please try again shortly.")
            return {"messages": [fallback], "tool_retry_count": 0}
        wait_time = 2 ** retry_count
        print(f"⚠️ Rate limited — waiting {wait_time}s before retry {retry_count + 1}/{max_retries}.")
        time.sleep(wait_time)
        return agent_node({**state, "tool_retry_count": retry_count + 1})

    except APIConnectionError as e:
        if retry_count >= max_retries:
            fallback = AIMessage(content="I'm having trouble connecting to required services right now. Please try again shortly.")
            return {"messages": [fallback], "tool_retry_count": 0}
        print(f"⚠️ Connection error — retrying ({retry_count + 1}/{max_retries}).")
        time.sleep(1)
        return agent_node({**state, "tool_retry_count": retry_count + 1})

    except BadRequestError as e:
        if "tool_use_failed" not in str(e):
            raise

        if retry_count >= max_retries:
            print(f"⚠️ Gave up after {max_retries} malformed tool-call retries.")
            fallback = AIMessage(content=(
                "I encountered repeated formatting errors while trying to use my tools for this "
                "request. I'm unable to complete this specific task right now — could you try "
                "rephrasing your question?"
            ))
            return {"messages": [fallback], "tool_retry_count": 0}

        print(f"⚠️ Malformed tool call (attempt {retry_count + 1}/{max_retries}) — retrying.")
        return agent_node({**state, "tool_retry_count": retry_count + 1})

    except Exception as e:
        import traceback
        print(f"⚠️⚠️⚠️ UNEXPECTED ERROR in agent_node: {type(e).__name__}: {e}")
        print("⚠️⚠️⚠️ If this is an AuthenticationError, check your API key immediately.")
        traceback.print_exc()
        fallback = AIMessage(content=(
            "I ran into an unexpected internal error while processing this request. "
            "Please try again — if this keeps happening, the issue has been logged for review."
        ))
        return {"messages": [fallback], "tool_retry_count": 0}

tool_node = ToolNode([tavily_search, search_documents])


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "dina-test-3"}}

    print("=== TURN 1 (conversational, should trigger NO_PLAN_NEEDED) ===")
    result1 = app.invoke(
        {"messages": [{"role": "user", "content": "My name is Dina and I'm building an agentic assistant."}]},
        config=config
    )
    print(f"Plan: {result1.get('plan')}")
    result1["messages"][-1].pretty_print()

    print("\n=== TURN 2 (should answer from history, NO tool call) ===")
    result2 = app.invoke(
        {"messages": [{"role": "user", "content": "What's my name and what am I building?"}]},
        config=config
    )
    print(f"Plan: {result2.get('plan')}")
    print(f"Total messages in state: {len(result2['messages'])}")
    for m in result2["messages"]:
        m.pretty_print()

    print("\n=== TURN 3 (possessive question NOT already in history, should route to search_documents) ===")
    result3 = app.invoke(
        {"messages": [{"role": "user", "content": "What tech stack did I use for my SwiftChat project?"}]},
        config=config
    )
    print(f"Plan: {result3.get('plan')}")
    result3["messages"][-1].pretty_print()