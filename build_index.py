"""
Build the Chroma vector store from the source PDFs.

Run this ONCE, locally, before deploying:

    python build_index.py

It writes a persistent Chroma DB to ./chroma_db — commit that folder (or
otherwise ship it) alongside app.py and rag.py. The deployed API only
*queries* this store; it never re-embeds the PDFs, so it never needs
pypdf or the source documents at runtime.
"""

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

PDF_PATHS = {
    "mental1.pdf": "/content/drive/MyDrive/mental heathy/mental1.pdf",
    "mental2.pdf": "/content/drive/MyDrive/mental heathy/mental2.pdf",
    "mental3.pdf": "/content/drive/MyDrive/mental heathy/mental3.pdf",
}

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mental_health"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def load_documents(pdf_paths: dict) -> list[dict]:
    """Extract and normalize text from each PDF."""
    documents = []

    for source, path in pdf_paths.items():
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        text = " ".join(text.replace("\n", " ").split())
        documents.append({"source": source, "text": text})
        print(f"{source}: {len(text)} characters")

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split each document's text into overlapping chunks."""
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "source": source})
            start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks total")
    return chunks


def build_chroma_collection() -> None:
    # DefaultEmbeddingFunction runs all-MiniLM-L6-v2 via ONNX runtime —
    # no torch, no sentence-transformers required.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    documents = load_documents(PDF_PATHS)
    chunks = chunk_documents(documents)

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    print(f"Added {len(chunks)} chunks to Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_chroma_collection()
