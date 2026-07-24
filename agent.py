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


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools([tavily_search, search_documents])


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful research assistant. You have access to two tools: "
    "'tavily_search' for current events, news, or general public web information, and "
    "'search_documents' for anything about Dina's own projects, resume, or personal documents.\n\n"
    "IMPORTANT: If the question refers to something personal or possessive — 'my project', "
    "'the SwiftChat project', 'her resume', 'this document' — ALWAYS use 'search_documents' "
    "FIRST, even if the name might also match something public on the web. Do not use "
    "'tavily_search' for anything that sounds like it belongs to Dina personally, since public "
    "results with a similar or identical name could be about a completely different, unrelated "
    "project — reporting those as fact would be a serious error.\n\n"
    "You are not required to use any tool — answer directly from your own knowledge whenever "
    "you're confident and the question doesn't require current information or document lookup. "
    "When you do search, formulate specific, keyword-focused queries rather than "
    "echoing the user's full question."
))


def planner_node(state: AgentState):
    user_question = state["messages"][-1].content

    planning_prompt = f"""Break the following research task into a short numbered list of concrete 
steps needed to answer it. If the task is simple enough to answer in one step, say so plainly — 
do not invent unnecessary steps for simple questions.

Task: {user_question}

Plan:"""

    plan_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    result = plan_llm.invoke(planning_prompt)

    return {"plan": result.content}


from groq import BadRequestError

def agent_node(state: AgentState):
    plan_context = SystemMessage(content=f"Here is the plan for this task:\n{state.get('plan', 'No plan yet.')}")
    base_messages = [SYSTEM_PROMPT, plan_context] + state["messages"]

    retry_count = state.get("tool_retry_count", 0)
    max_retries = 2

    try:
        response = llm_with_tools.invoke(base_messages)
        return {"messages": [response], "tool_retry_count": 0}

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

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"messages": [{"role": "user", "content": "Look up what backend framework and real-time technology I used for SwiftChat, then search the web for the most common security vulnerabilities specific to that exact combination."}]})
    print("=== PLAN ===")
    print(result.get("plan", "no plan generated"))
    print("=== END PLAN ===\n")
    for m in result["messages"]:
        m.pretty_print()