# 3GPP RAG Chatbot — Mavenir GET Assignment

A Retrieval-Augmented Generation (RAG) chatbot built on 3GPP Telecom Standards
documentation, designed for **near-zero hallucinations**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | NVIDIA Nemotron (OpenAI-compatible API) |
| Embeddings | BAAI/bge-small-en-v1.5 (Sentence Transformers) |
| Vector Store | ChromaDB (persistent, local) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |

---

## Architecture

```
User Query
    ↓
Embedder (BAAI/bge-small-en-v1.5)
    ↓
ChromaDB — Semantic Retrieval (Top-K chunks)
    ↓
Hallucination Guard (similarity threshold filter)
    ↓
NVIDIA Nemotron LLM (context-grounded answer)
    ↓
Answer + Confidence Score + Source Citation
```

---

## Anti-Hallucination Strategy

- Similarity threshold — chunks scoring below **0.40** are blocked
- LLM is restricted to retrieved context only via system prompt
- Source citation enforced in every response
- Confidence scoring: `high` / `medium` / `low` / `none`
- Qwen chain-of-thought `<think>` blocks stripped before grounding check

---

## Knowledge Base

| File | Specification |
|---|---|
| `23501-k20.docx` | TS 23.501 — 5G System Architecture |
| `21915-f00.docx` | TS 21.915 — Release 15 Description |

---

## Project Structure

```
CHATBOT/
├── main.py               ← FastAPI server
├── ingest.py             ← Run once to populate ChromaDB
├── requirements.txt      ← All dependencies
├── Dockerfile            ← For deployment
├── .env                  ← API keys (not committed)
│
├── rag/
│   ├── loader.py         ← Load .docx 3GPP spec files
│   ├── chunker.py        ← Recursive text splitting (chunk=800)
│   ├── embedder.py       ← Sentence Transformers embeddings
│   ├── vectorstore.py    ← ChromaDB store and retrieve
│   ├── retriever.py      ← Semantic retrieval with threshold
│   ├── generator.py      ← LLM prompt engineering + call
│   └── hallucination.py  ← Confidence guard layer
│
├── static/
│   └── index.html        ← Chat UI
│
└── docs/
    ├── 23501-k20.docx
    └── 21915-f00.docx
```

---

## How to Run Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set up environment — create a `.env` file**
```
NVIDIA_API_KEY=your_key_here
```

**3. Ingest documents into ChromaDB (run only once)**
```bash
python ingest.py
```

**4. Start the server**
```bash
uvicorn main:app --reload --port 8000
```

**5. Open in browser**
```
http://localhost:8000
```

---

## Candidate

**Mayank Chaudhary**
Submitted for: Graduate Engineer Trainee (GET) — Mavenir Systems
