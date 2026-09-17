"""
MindPower RAG core — deployment-light version.

Retrieval: Chroma with its default ONNX embedding function (no torch).
Generation: Google Gemini via google-generativeai (HTTP call,
no local model weights, no torch/transformers installed).

If the generation call fails (model unavailable, rate-limited, timed
out), answer_question() falls back to an extractive answer built
directly from the retrieved chunks, so the response is still grounded
in the knowledge base instead of a bare error.
"""

import logging
import os

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError

logger = logging.getLogger("mindpower.rag")

CHROMA_DIR = "chromadb"
COLLECTION_NAME = "mentalhealth"

# Override with an env var if you switch to a different Gemini model.
# Note: there is no "Gemini 3.5" model — the line runs 1.0 / 1.5 / 2.0 / 2.5.
# gemini-2.5-flash is a solid default: fast and inexpensive for grounded Q&A.
GEN_MODEL_NAME = os.environ.get("MINDPOWER_GEN_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

SYSTEM_PROMPT = """You are a mental health support assistant.
Answer using the provided knowledge only.
Do not diagnose mental illnesses.
If the provided knowledge does not contain enough information,
say that you do not have enough information.
If the user's message suggests they may be in crisis or at risk of harming
themselves, gently encourage them to contact a crisis line or emergency
services in their area in addition to anything else you say."""

NO_RESULTS_MESSAGE = (
    "I don't have enough information in my knowledge base to answer that "
    "right now. If you're comfortable, could you tell me a bit more about "
    "what you're experiencing?"
)


def get_collection():
    """Open the pre-built Chroma store (see build_index.py)."""
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    count = collection.count()
    if count == 0:
        logger.warning(
            "Chroma collection '%s' at %s is EMPTY. Every question will "
            "return the 'not enough information' fallback until this is "
            "fixed. Make sure build_index.py was run and that the "
            "chroma_db/ folder was actually deployed alongside app.py.",
            COLLECTION_NAME,
            CHROMA_DIR,
        )
    else:
        logger.info("Loaded Chroma collection '%s' with %d chunks.", COLLECTION_NAME, count)

    return collection


def get_generator_client() -> genai.Client:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. Create a key at "
            "https://aistudio.google.com/app/apikey and set it in your "
            "Render environment variables."
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def retrieve(collection, query: str, k: int = 2) -> list[str]:
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


def build_fallback_answer(context_chunks: list[str]) -> str:
    """
    Extractive fallback used when the generation model is unavailable.
    Returns the most relevant retrieved passage(s) directly, framed as
    reference material rather than a conversational reply, plus a note
    that this is unedited source text.
    """
    if not context_chunks:
        return NO_RESULTS_MESSAGE

    excerpt = context_chunks[0].strip()
    if len(excerpt) > 700:
        excerpt = excerpt[:700].rsplit(" ", 1)[0] + "…"

    return (
        f"\"{excerpt}\"\n\n"
        "(This is a direct excerpt from the knowledge base — generation was "
        "unavailable, so no conversational answer could be composed.)"
    )


def answer_question(
    question: str,
    collection,
    client: genai.Client,
    k: int = 2,
    max_tokens: int = 200,
) -> str:
    context_chunks = retrieve(collection, question, k=k)

    if not context_chunks:
        return NO_RESULTS_MESSAGE

    context = "\n\n".join(context_chunks)

    prompt = (
        f"Knowledge:\n\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Give a clear, supportive answer based on the knowledge above."
    )

    try:
        response = client.models.generate_content(
            model=GEN_MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    except (APIError, TimeoutError, Exception) as exc:
        # Broad catch is intentional here: the Gemini client can raise
        # several different exception types (API errors, timeouts,
        # connection errors) depending on the failure mode, and all of them
        # should degrade to the same RAG-grounded fallback rather than a 500.
        logger.warning("Generation call failed, falling back to retrieval: %s", exc)
        return build_fallback_answer(context_chunks)
