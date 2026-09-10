"""
MindPower API
-------------
FastAPI wrapper around the RAG pipeline in mental_health_rag.py.

Run with:
    uvicorn app:app --reload --port 8000

Then open mindpower.html (or serve it from this same app — see the
StaticFiles mount below) and it will call POST /api/ask automatically.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from mental_health_rag import (
    EMBED_MODEL_NAME,
    answer_question,
    build_generator,
    load_index,
)

# ---------------------------------------------------------------------------
# Loaded once at startup, reused across requests (loading the models per
# request would be far too slow).
# ---------------------------------------------------------------------------

ml_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
    ml_state["index"], ml_state["chunks"] = load_index()
    ml_state["generator"] = build_generator()
    yield
    ml_state.clear()


app = FastAPI(title="MindPower API", lifespan=lifespan)

# Allow the HTML front end (served from another origin, or opened as a
# local file) to call this API. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = answer_question(
            question,
            ml_state["generator"],
            ml_state["index"],
            ml_state["chunks"],
            ml_state["embed_model"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return AskResponse(answer=answer)


# Optional: serve mindpower.html directly from this server at "/" so you
# don't need a separate static file server. Place mindpower.html in the
# same folder as this file (or point StaticFiles at another directory).
app.mount("/", StaticFiles(directory=".", html=True), name="static")
