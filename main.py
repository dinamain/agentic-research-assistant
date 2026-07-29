from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import app as agent_app

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    thread_id: str


async def stream_agent_response(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    for message_chunk, metadata in agent_app.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "agent" and message_chunk.content:
            yield f"data: {message_chunk.content}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_agent_response(request.message, request.thread_id),
        media_type="text/event-stream"
    )