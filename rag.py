"""
MindPower RAG core — deployment-light version.

Retrieval: Chroma with its default ONNX embedding function (no torch).
Generation: Hugging Face Inference API via huggingface_hub (HTTP call,
no local model weights, no torch/transformers installed).
"""

import os

import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mental_health"

# Override with an env var if you switch to a different hosted model.
GEN_MODEL_NAME = os.environ.get("MINDPOWER_GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = """You are a mental health support assistant.
Answer using the provided knowledge only.
Do not diagnose mental illnesses.
If the provided knowledge does not contain enough information,
say that you do not have enough information.
If the user's message suggests they may be in crisis or at risk of harming
themselves, gently encourage them to contact a crisis line or emergency
services in their area in addition to anything else you say."""


def get_collection():
    """Open the pre-built Chroma store (see build_index.py)."""
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def get_generator_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. Create a token at "
            "https://huggingface.co/settings/tokens and set it in your "
            "Render environment variables."
        )
    return InferenceClient(model=GEN_MODEL_NAME, token=HF_TOKEN)


def retrieve(collection, query: str, k: int = 2) -> list[str]:
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


def answer_question(
    question: str,
    collection,
    client: InferenceClient,
    k: int = 2,
    max_tokens: int = 200,
) -> str:
    context_chunks = retrieve(collection, question, k=k)
    context = "\n\n".join(context_chunks)

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

    response = client.chat_completion(messages=messages, max_tokens=max_tokens)
    return response.choices[0].message.content
