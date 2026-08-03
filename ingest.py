from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os
import re

CHROMA_DIR = "./chroma_db"


def clean_text(text: str) -> str:
    text = re.sub(r'(\w)-?\n(\w)', r'\1\2', text)
    return text


def ingest_pdf(pdf_path: str, vectorstore=None, original_filename: str = None):
    filename = original_filename or os.path.basename(pdf_path)
    
    if vectorstore is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    existing = vectorstore.get(where={"filename": filename})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        print(f"Removed {len(existing['ids'])} old chunks for {filename}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    for doc in documents:
        doc.metadata["filename"] = filename
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        page = chunk.metadata.get("page", "unknown")
        header = f"[Source: {filename}, page {page}]\n"
        chunk.page_content = header + chunk.page_content

    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError(f"No usable content extracted from {filename} — the file may be corrupted or not a valid PDF.")

    vectorstore.add_documents(chunks)
    print(f"Done! {filename} ingested successfully.")