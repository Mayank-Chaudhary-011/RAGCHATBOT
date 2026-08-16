from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.hallucination import check_hallucination_risk

load_dotenv()

# In-memory cache: normalized question -> full response dict
_response_cache: dict[str, dict] = {}

app = FastAPI(title="3GPP RAG Chatbot — Mavenir")
app.mount("/static", StaticFiles(directory="static"), name="static")

embedder = Embedder()
vector_store = VectorStore()
retriever = Retriever(embedder, vector_store)
generator = Generator()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/chat")
def chat(request: QueryRequest):
    cache_key = request.question.strip().lower()

    # Return cached answer if the same question was asked before
    if cache_key in _response_cache:
        cached = _response_cache[cache_key].copy()
        cached["cached"] = True
        return cached

    docs = retriever.retrieve(request.question, top_k=request.top_k)
    result = generator.generate(request.question, docs)
    guarded = check_hallucination_risk(docs, result["answer"])
    response = {
        "question": request.question,
        "answer": guarded["answer"],
        "confidence": guarded["confidence"],
        "grounded": guarded["grounded"],
        "sources": guarded["sources"],
        "retrieved_count": len(docs),
        "cached": False
    }
    _response_cache[cache_key] = response
    return response


@app.get("/health")
def health():
    return {"status": "ok", "docs_in_store": vector_store.count()}