# chromadb-orm

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-E44D26)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--mpnet--base--v2-FF6F00)
![LangChain](https://img.shields.io/badge/LangChain-text%20splitting-1C3C3C)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%2FCPU%2FMPS-EE4C2C?logo=pytorch&logoColor=white)

**A production-ready RAG backend — ingest PDFs, DOCX, and plain text; embed with a local transformer; store in ChromaDB; retrieve the most relevant chunks for any query.**

---

## What this does

Most RAG demos dump every document into a single flat collection. This system is different: it organises documents into *semantically grouped collections* using centroid-based similarity, so retrieval scales cleanly as your document library grows.

```
Upload file (PDF / DOCX / TXT)
        |
        v
  Text extraction
  (pdfplumber / python-docx / plain read)
        |
        v
  Recursive chunking
  (LangChain, chunk=1000 / overlap=150)
        |
        v
  Dense embedding
  (SentenceTransformer: all-mpnet-base-v2)
        |
        v
  Collection routing
  +------------------------------------------+
  |  Compute batch centroid embedding        |
  |  Compare to every collection centroid    |
  |  cosine similarity >= 0.35 -> assign     |
  |  no match -> create new collection       |
  +------------------------------------------+
        |
        v
  ChromaDB (HTTP, localhost:8000)
  Stores: text chunks + embeddings + metadata
  Special doc per collection: "centroid"
        |
  ------+------
  |             |
  v             v
UPLOAD       SEARCH
/upload      /search
             |
             v
      Query is chunked + embedded
      -> centroid compared to all collections
      -> top-k collections selected
      -> top-k documents retrieved per collection
      -> chunks returned as LLM context
```

Pass the returned chunks to any LLM (OpenAI, Anthropic, local Ollama) as context — `chromadb-orm` is intentionally model-agnostic.

---

## Architecture

| Layer | Technology | Role |
|---|---|---|
| API | FastAPI + async | Exposes `/upload`, `/search`, `/health_db` |
| Text extraction | pdfplumber, python-docx | Handles PDF, DOCX, TXT |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | 1000 chars, 150 overlap |
| Embedding | `all-mpnet-base-v2` (SentenceTransformers) | 768-d dense vectors, GPU/MPS/CPU |
| Vector store | ChromaDB (HTTP mode) | Persistent; manages collections + centroids |
| Collection routing | Custom cosine-similarity logic (PyTorch) | Auto-clusters documents by topic |
| Retrieval | ChromaDB `collection.query` | Returns top-k chunks for LLM context |

### Collection-centroid routing

Every group of semantically related documents is stored together. Each collection maintains a `centroid` document — the mean embedding of all chunks. On upload, each batch of chunks is compared to every centroid; the closest collection above threshold (0.35) wins, otherwise a new collection is created. Queries hit only the relevant collections, keeping latency low even with hundreds of documents.

---

## Project structure

```
chromadb-orm/
├── main.py                    # FastAPI entrypoint (port 9000)
├── requirements.txt
├── app/
│   ├── api/
│   │   ├── routes.py          # /upload  /search  /health_db
│   │   └── request.py         # Dataclasses: UploadDocument, SearchDocument
│   ├── db/
│   │   └── client.py          # ChromaDB client + centroid logic + top-k retrieval
│   ├── document/
│   │   ├── extract.py         # Text extraction + embedding model + chunking
│   │   └── batch.py           # Async pipeline: process_file -> process_text -> process_embeddings
│   └── logger.py              # Structured logger (console + file)
└── test/
    └── docs/                  # sample.pdf, sample.docx, sample.txt for local testing
```

---

## Setup

### Prerequisites

- Python 3.10+
- ChromaDB server running on `localhost:8000`

```bash
pip install chromadb
chroma run --path ./chroma_data --port 8000
```

### Install

```bash
git clone https://github.com/sarvesh1karandikar/chromadb.git
cd chromadb
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

The server clears all collections on startup (clean-slate dev mode). To persist data across restarts, remove the `lifespan` function in `main.py`.

---

## API reference

### `POST /upload`

Upload one or more files. Returns chunk counts, embedding shape, and assigned collection names.

```bash
curl -X POST http://localhost:9000/upload \
  -F "files=@test/docs/sample.pdf" \
  -F "files=@test/docs/sample.txt"
```

### `POST /search/`

Search across all collections. Returns the top-k most relevant text chunks.

```bash
curl -X POST http://localhost:9000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k_collections": 2, "top_k_documents": 5}'
```

Response includes `top_k_results` — an array of raw text chunks ready to inject into an LLM prompt.

### `GET /health_db`

Returns ChromaDB connectivity status.

---

## Live demo

What a working demo looks like:

1. Upload a product manual (PDF) and a policy document (DOCX) via `/upload`
2. Ask "What is the warranty period?" via `/search/`
3. The API returns the 5 most relevant chunks
4. Feed those chunks + the question to an LLM:
   `You are a helpful assistant. Context: {chunks}. Question: {query}`
5. Get a grounded, citation-ready answer

The included test docs (`test/docs/`) let you reproduce this locally in under 2 minutes.

---

## What I learned

- **Centroid-based collection routing** — treating the mean embedding of a collection as a routing key is a lightweight alternative to a full hierarchical index. Documents cluster naturally; very different ones get their own collection.
- **Async all the way down** — embedding generation (`asyncio.to_thread`) and ChromaDB writes are non-blocking, which matters when uploading 10+ documents concurrently.
- **Chunking strategy matters** — 1000-char chunks with 150-char overlap retained enough context for multi-sentence questions while keeping embedding count manageable.
- **Hardware-aware device selection** — the model loader detects CUDA then MPS (Apple Silicon) then CPU automatically, so the same code runs on a MacBook Pro and a GPU server without changes.

---

## Part of a 3-repo system

| Repo | Status | Role |
|---|---|---|
| **chromadb-orm** (this repo) | Active | RAG backend: document ingestion, vector storage, retrieval |
| **llm-server** | Planned | Kubernetes-hosted LLM service (Ollama / vLLM) |
| **ui** | Planned | Frontend: drag-and-drop PDF upload, chat interface |

The design is intentionally decoupled — swap ChromaDB for Pinecone or Weaviate, or point the `/search` output at any LLM endpoint, without touching the other layers.

---

## Contributing

Issues and PRs welcome. To add a new file format, implement `extract_<format>(file_obj)` in `app/document/extract.py` and extend the `extract_text` dispatcher.
