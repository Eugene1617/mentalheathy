"""
Mental Health Support RAG Pipeline
----------------------------------
Pipeline: PDF ingestion -> chunking -> embedding/indexing -> retrieval -> generation.

Requires: pypdf, sentence-transformers, faiss-cpu, transformers
"""

import pickle

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

'''PDF_PATHS = {
    "mental1.pdf": "mental1.pdf",
    "mental2.pdf": "mental2.pdf",
    "mental3.pdf": "mental3.pdf",
}'''

INDEX_PATH = "mental_health.index"
CHUNKS_PATH = "mental_chunks.pkl"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are a mental health support assistant.
Answer using the provided knowledge only.
Do not diagnose mental illnesses.
If the provided knowledge does not contain enough information,
say that you do not have enough information.
If the user's message suggests they may be in crisis or at risk of harming
themselves, gently encourage them to contact a crisis line or emergency
services in their area in addition to anything else you say."""


# ---------------------------------------------------------------------------
# 1. Load and clean PDFs
# ---------------------------------------------------------------------------

'''def load_documents(pdf_paths: dict) -> list[dict]:
    """Extract and normalize text from each PDF."""
    documents = []

    for source, path in pdf_paths.items():
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        # Normalize whitespace
        text = " ".join(text.replace("\n", " ").split())

        documents.append({"source": source, "text": text})
        print(f"{source}: {len(text)} characters")

    return documents'''


# ---------------------------------------------------------------------------
# 2. Chunk documents
# ---------------------------------------------------------------------------

'''def chunk_documents(
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
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": source})
            start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks total")
    return chunks'''


# ---------------------------------------------------------------------------
# 3. Build (or load) the FAISS index
# ---------------------------------------------------------------------------

'''def build_index(chunks: list[dict], embed_model: SentenceTransformer) -> faiss.Index:
    """Embed all chunks and build a FAISS index over them."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index'''


'''def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)'''


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    k: int = 2,
) -> list[dict]:
    """Return the top-k most relevant chunks for a query."""
    query_embedding = embed_model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------

def build_generator(model_name: str = GEN_MODEL_NAME):
    return pipeline("text-generation", model=model_name)


def answer_question(
    question: str,
    generator,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    k: int = 2,
    max_new_tokens: int = 80,
) -> str:
    """Full RAG step: retrieve context, then generate a grounded answer."""
    results = retrieve(question, index, chunks, embed_model, k=k)
    context = "\n\n".join(r["text"] for r in results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Knowledge:\n\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Give a clear, supportive answer based on the knowledge above."
            ),
        },
    ]

    response = generator(messages, max_new_tokens=max_new_tokens)
    return response[0]["generated_text"][-1]["content"]

