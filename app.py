"""
MindPower API — deployment-light version.

No torch, no transformers, no faiss, no sentence-transformers.
Retrieval runs on Chroma (ONNX embeddings); generation calls Google
Gemini over HTTP via google-genai.

Run locally with:
    uvicorn app:app --reload --port 8000

On Render: set GOOGLE_API_KEY in the environment, start command
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import answer_question, get_collection, get_generator_client

state: dict = {}

# One shared limiter, keyed by client IP. /api/ask is the expensive path
# (it calls the Gemini API), so it gets a tighter cap than the rest of
# the app. Adjust the rate string ("10/minute") to taste — slowapi accepts
# "N/second", "N/minute", "N/hour", "N/day".
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["collection"] = get_collection()
    state["client"] = get_generator_client()
    yield
    state.clear()


app = FastAPI(title="MindPower API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute")
def ask(request: Request, payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # answer_question() already falls back to an extractive,
        # retrieval-grounded answer if the generation model fails —
        # this except only fires for retrieval/infrastructure errors
        # (e.g. the Chroma store itself being unreachable).
        answer = answer_question(question, state["collection"], state["client"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    return AskResponse(answer=answer)


# Serves mindpower.html directly from this server at "/".
app.mount("/", StaticFiles(directory=".", html=True), name="static")
