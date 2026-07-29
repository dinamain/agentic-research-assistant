from agent import app

config = {"configurable": {"thread_id": "dina-invoke-vs-stream-1"}}

print("=== .invoke() ===")
result = app.invoke(
    {"messages": [{"role": "user", "content": "What tech stack did I use for SwiftChat"}]},
    config=config
)
result["messages"][-1].pretty_print()