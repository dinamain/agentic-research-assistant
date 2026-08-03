from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import app as agent_app
from fastapi import UploadFile, File
import shutil
import os
from ingest import ingest_pdf
import tempfile
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://agentic-assistant-frontend.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Agentic Research Assistant API is running"}



from retriever_tool import vectorstore as shared_vectorstore

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        ingest_pdf(temp_path, vectorstore=shared_vectorstore, original_filename=file.filename)
    finally:
        os.remove(temp_path)

    return {"message": f"{file.filename} ingested successfully"}
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