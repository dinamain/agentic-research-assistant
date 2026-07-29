from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os

CHROMA_DIR = "./chroma_db"

_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
_vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
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

        # NOTE: cross-encoder re-ranking removed here (was sentence-transformers/torch-based —
        # a full PyTorch install, on top of FastEmbed's ONNX embeddings) to fix a Render free-tier
        # OOM. Falling back to vector-similarity-with-score directly from ChromaDB, filtered by a
        # distance threshold, instead of cross-encoder relevance scoring. Lower precision than
        # re-ranking, but a deliberate memory/precision tradeoff for this deployment tier — the
        # RAG Document Q&A project (github.com/dinamain/rag-document-qa) has the full re-ranking
        # pipeline and documents this same tradeoff in more depth.
        results = _vectorstore.similarity_search_with_score(rewritten, k=5)

        if not results:
            return "No relevant documents found in the document store."

        relevant = [chunk for chunk, score in results if score < 0.87]

        if not relevant:
            return "No sufficiently relevant document content found for this query."

        return "\n\n---\n\n".join(chunk.page_content for chunk in relevant)

    except Exception as e:
        print(f"⚠️ search_documents failed internally: {e}")
        return f"Document search encountered an error and could not complete: {e}"