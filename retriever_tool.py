from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os

CHROMA_DIR = "./chroma_db"

_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
_vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
_rewrite_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)


def _rewrite_query(question: str) -> str:
    prompt = f"""Rewrite the following question into a clear, specific search query 
optimized for semantic search over documents. Keep it concise. 
Return ONLY the rewritten query, nothing else.

Question: {question}

Rewritten query:"""
    result = _rewrite_llm.invoke(prompt)
    return result.content.strip().strip('"')


@tool
def search_documents(query: str) -> str:
    """Search through Dina's stored documents (resume, books, notes) for relevant information.
    Use this for questions about specific documents Dina has uploaded — NOT for current events,
    news, or general knowledge questions that don't reference a specific stored document."""

    try:
        rewritten = _rewrite_query(query)

        retriever = _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        candidates = retriever.invoke(rewritten)

        if not candidates:
            return "No relevant documents found in the document store."

        pairs = [[query, chunk.page_content] for chunk in candidates]
        scores = _reranker.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        relevant = [chunk for chunk, score in scored if score > -3.0][:5]

        if not relevant:
            return "No sufficiently relevant document content found for this query."

        return "\n\n---\n\n".join(chunk.page_content for chunk in relevant)

    except Exception as e:
        print(f"⚠️ search_documents failed internally: {e}")
        return f"Document search encountered an error and could not complete: {e}"