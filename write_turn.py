from agent import app

config = {"configurable": {"thread_id": "user-dina-persist-test"}}
result = app.invoke(
    {"messages": [{"role": "user", "content": "My favorite number is 47."}]},
    config=config
)
result["messages"][-1].pretty_print()