"""
MindPower API — deployment-light version.

No torch, no transformers, no faiss, no sentence-transformers.
Retrieval runs on Chroma (ONNX embeddings); generation calls the
Hugging Face Inference API over HTTP.

Run locally with:
    uvicorn app:app --reload --port 8000

On Render: set HF_TOKEN in the environment, start command
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import answer_question, get_collection, get_generator_client

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["collection"] = get_collection()
    state["client"] = get_generator_client()
    yield
    state.clear()


app = FastAPI(title="MindPower API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = answer_question(question, state["collection"], state["client"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return AskResponse(answer=answer)


# Serves mindpower.html directly from this server at "/".
app.mount("/", StaticFiles(directory=".", html=True), name="static")
