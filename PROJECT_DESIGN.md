# Project Design — 3GPP RAG Chatbot

## 1. Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions on 3GPP Telecom Standards with near-zero hallucinations. The system retrieves relevant passages from official 3GPP specification documents, validates the retrieved context against a confidence threshold, and generates grounded answers — refusing to respond when sufficient context is unavailable.

## 2. Knowledge Base

| Document | Specification | Content |
|---|---|---|
| `23501-k20.docx` | TS 23.501 | 5G System Architecture |
| `21915-f00.docx` | TS 21.915 | Release 15 Description |

## 3. System Architecture

```
                    ┌──────────────────────────┐
                    │      User Interface       │
                    │   (static/index.html)     │
                    └────────────┬─────────────┘
                                 │ POST /chat
                                 ▼
                    ┌──────────────────────────┐
                    │     FastAPI Server        │
                    │       (main.py)           │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
     ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
     │   Retriever     │ │   Generator   │ │  Hallucination   │
     │ (retriever.py)  │ │(generator.py) │ │     Guard        │
     └───────┬────────┘ └───────────────┘ │(hallucination.py)│
             │                             └──────────────────┘
     ┌───────┴────────┐
     │   Embedder      │
     │  (embedder.py)  │
     └───────┬────────┘
             │
     ┌───────┴────────┐
     │   ChromaDB      │
     │(vectorstore.py) │
     └────────────────┘
```

## 4. Data Flow

### 4.1 Ingestion Pipeline (run once via `ingest.py`)

```
.docx Files → Loader → Chunker → Embedder → ChromaDB
```

1. **Loader** — Reads `.docx` files, extracts text from paragraphs and tables
2. **Chunker** — Splits into 800-character chunks with 200-character overlap using recursive text splitting
3. **Embedder** — Generates 4096-dimensional embeddings via NVIDIA API (`nv-embedqa-e5-v5`)
4. **VectorStore** — Stores chunks with embeddings and metadata in ChromaDB (cosine distance)

**Result:** 5,237 indexed chunks across both documents.

### 4.2 Query Pipeline (every user question)

```
User Query → Embed Query → Retrieve Top-5 → Hallucination Guard → LLM Generation → Response
```

1. **Embed** — User query is embedded using the same NVIDIA model with `input_type: "query"`
2. **Retrieve** — ChromaDB returns top-5 chunks by cosine similarity, filtered at threshold ≥ 0.35
3. **Hallucination Guard** — Blocks response if top similarity score < 0.40
4. **Generate** — NVIDIA Nemotron LLM generates an answer constrained to retrieved context only
5. **Response** — Returns answer with confidence level, grounded status, and source document citations

## 5. Component Design

### 5.1 Document Loader (`rag/loader.py`)

- Parses `.docx` files using `python-docx`
- Extracts both paragraph text and table content (3GPP specs contain significant data in tables)
- Handles import collision between `python-docx.Document` and `langchain.Document` via aliasing

### 5.2 Chunker (`rag/chunker.py`)

- Uses `RecursiveCharacterTextSplitter` from LangChain
- Chunk size: **800 characters**, Overlap: **200 characters**
- Split hierarchy: `\n\n` → `\n` → `. ` → ` ` → character-level
- Preserves metadata (source filename) through splitting

### 5.3 Embedder (`rag/embedder.py`)

- Model: **NVIDIA nv-embedqa-e5-v5** (4096 dimensions)
- Asymmetric retrieval: uses `input_type: "passage"` for documents, `input_type: "query"` for user questions
- Batched API calls (10 texts per request) for efficient ingestion

### 5.4 Vector Store (`rag/vectorstore.py`)

- Persistent ChromaDB storage in `chroma_db/` directory
- Distance metric: **Cosine** (`hnsw:space: cosine`)
- Each chunk stored with UUID, text, embedding, and source metadata

### 5.5 Retriever (`rag/retriever.py`)

- Converts cosine distance to similarity score: `score = 1 - distance`
- Retrieves top-5 chunks above **0.35 similarity threshold**
- Returns structured results with text, metadata, and score

### 5.6 Generator (`rag/generator.py`)

- Model: **NVIDIA Nemotron** (`nvidia/nemotron-3.5-lightning-30b-a3b`)
- Temperature: **0.1** (near-deterministic for factual accuracy)
- System prompt enforces:
  - Answer only from provided context
  - Cite source document in every response
  - Return exact fallback message when context is insufficient
  - Never fabricate specifications or procedures
- Strips `<think>` chain-of-thought blocks from output via regex

### 5.7 Hallucination Guard (`rag/hallucination.py`)

Three-tier validation:

| Condition | Response | Confidence |
|---|---|---|
| No documents retrieved | Fallback message | `none` |
| Top similarity score < 0.40 | Fallback message | `low` |
| Top score ≥ 0.40 | LLM-generated answer | `high` (≥0.50) / `medium` (<0.50) |

Additionally checks if the LLM output contains the fallback phrase — if so, marks `grounded: false`.

## 6. Anti-Hallucination Strategy

Five layers of protection working together:

1. **Retrieval threshold (0.35)** — Irrelevant chunks are filtered before reaching the LLM
2. **Hallucination guard threshold (0.40)** — Low-confidence retrievals produce a safe fallback instead of an answer
3. **System prompt restriction** — LLM is constrained to answer only from the provided context
4. **Low temperature (0.1)** — Minimizes randomness in generation
5. **Chain-of-thought stripping** — Removes internal reasoning blocks that could interfere with grounding validation

## 7. API Design

### `POST /chat`

**Request:**
```json
{
  "question": "What is the role of AMF in 5G?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is the role of AMF in 5G?",
  "answer": "According to 23501-k20, AMF stands for Access and Mobility Management function...",
  "confidence": "high",
  "grounded": true,
  "sources": ["23501-k20.docx"],
  "retrieved_count": 5,
  "cached": false
}
```

### `GET /health`

Returns system status and document count in the vector store.

## 8. Project Structure

```
CHATBOT/
├── main.py                ← FastAPI server + /chat endpoint
├── ingest.py              ← One-time ingestion pipeline
├── requirements.txt       ← Python dependencies
├── Dockerfile             ← Container deployment
├── .env                   ← API keys (not committed)
│
├── rag/
│   ├── loader.py          ← .docx parser (paragraphs + tables)
│   ├── chunker.py         ← Recursive text splitter
│   ├── embedder.py        ← NVIDIA embedding API
│   ├── vectorstore.py     ← ChromaDB operations
│   ├── retriever.py       ← Semantic search + threshold filter
│   ├── generator.py       ← LLM prompt + generation
│   └── hallucination.py   ← Confidence guard
│
├── static/
│   └── index.html         ← Chat interface
│
├── docs/
│   ├── 23501-k20.docx     ← TS 23.501
│   └── 21915-f00.docx     ← TS 21.915
│
└── chroma_db/             ← Persistent vector store (auto-generated)
```

## 9. Configuration

| Parameter | Value | Purpose |
|---|---|---|
| Chunk size | 800 chars | Balance between context and precision |
| Chunk overlap | 200 chars | Preserve context across chunk boundaries |
| Embedding dimensions | 4,096 | High-fidelity semantic representation |
| Retrieval top-K | 5 | Number of candidate chunks per query |
| Retrieval threshold | 0.35 | Minimum similarity to consider a chunk relevant |
| Hallucination threshold | 0.40 | Minimum confidence to return an answer |
| LLM temperature | 0.1 | Near-deterministic for factual accuracy |
| Distance metric | Cosine | Scale-invariant, suited for high-dimensional vectors |

## 10. Deployment

- **Platform:** Render.com (free tier)
- **Runtime:** Python 3.10
- **Build:** `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Live URL:** https://ragchatbot-eidf.onrender.com
