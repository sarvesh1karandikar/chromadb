# CLAUDE.md — chromadb-orm

## Project summary

`chromadb-orm` is a FastAPI-based RAG (Retrieval-Augmented Generation) backend service. It provides a REST API for ingesting documents (PDF, DOCX, TXT), embedding them with a local SentenceTransformer model, and storing them in ChromaDB. A custom centroid-routing layer automatically clusters documents into semantically grouped collections, and a `/search` endpoint retrieves the top-k most relevant text chunks for any query.

The service is model-agnostic: it returns raw text chunks that the caller feeds to any LLM.

---

## How to run

### Dependencies

```bash
pip install -r requirements.txt
```

### External services required

- **ChromaDB HTTP server** must be running at `localhost:8000` before the FastAPI app starts:

```bash
chroma run --path ./chroma_data --port 8000
```

### Environment variables

No `.env` file is required. The following are hardcoded defaults you may want to override:

| Variable | Default | Where |
|---|---|---|
| ChromaDB host | `localhost` | `app/db/client.py` line 1 |
| ChromaDB port | `8000` | `app/db/client.py` line 1 |
| FastAPI port | `9000` | `main.py` |
| Embedding model | `all-mpnet-base-v2` | `app/document/extract.py` |
| Collection similarity threshold | `0.35` (upload), `0.25` (search) | `app/db/client.py` |

### Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

**Warning**: The `lifespan` function in `main.py` deletes all ChromaDB collections on startup. This is intentional for dev. Remove or guard it before production use.

---

## Architecture details

### Embedding model

- Model: `sentence-transformers/all-mpnet-base-v2`
- Dimension: 768
- Device: auto-selected — CUDA > MPS (Apple Silicon) > CPU
- Generation is wrapped in `asyncio.to_thread` to avoid blocking the event loop

### Chunking strategy

- Class: `LangChain RecursiveCharacterTextSplitter`
- `chunk_size=1000`, `chunk_overlap=150`
- Separators: `["\n\n", "\n", ".", " ", ""]`
- Applied to both uploaded documents and search queries

### Collection routing (centroid model)

Each collection stores a special document with `id="centroid"` containing the mean embedding of all non-centroid chunks. On upload:

1. A batch of chunks (size 5) is embedded
2. Their mean (centroid) is computed
3. Cosine similarity is computed against every existing collection centroid
4. If similarity >= 0.35: assign to that collection
5. If no match: create a new `collection-<uuid>` collection
6. After every batch insert, the collection centroid is recomputed

On search:

1. Query is chunked and embedded
2. Query centroid is computed
3. Top-k collections are selected (threshold 0.25, cosine similarity)
4. Top-k documents are retrieved from each selected collection using `collection.query`

### Pipeline flow

```
UploadFile -> extract_text -> text_splitter.split_text -> generate_embeddings
          -> top_1_collection (centroid routing) -> add_to_collection -> update_collection_centroid
```

---

## Current limitations

1. **No LLM integration** — the service returns raw text chunks. There is no built-in prompt assembly or LLM call. The caller must handle that.
2. **No authentication** — the API is fully open. Not safe to expose publicly without a proxy/auth layer.
3. **Startup wipe** — `main.py` deletes all collections on startup. Data does not persist across server restarts in dev mode.
4. **No streaming** — `/search` is a blocking request-response. No SSE or websocket support.
5. **Single-node ChromaDB** — uses HTTP client pointed at localhost; no support for distributed or cloud ChromaDB.
6. **No PDF OCR** — scanned/image-only PDFs will extract no text (PyPDFLoader only reads text layer).
7. **No document deduplication** — re-uploading the same file creates duplicate chunks.
8. **No metadata filtering** — retrieval cannot be scoped to a specific file or date range.

---

## Enhancement TODO

### Quick wins (< 1 day each)

- [ ] Move ChromaDB host/port and model name to environment variables / `.env`
- [ ] Add `PERSIST_ON_STARTUP=false` env flag to guard the collection-wipe in `lifespan`
- [ ] Return source filename metadata alongside retrieved chunks
- [ ] Add document deduplication (hash chunk content before insert)

### Medium lift (1–3 days each)

- [ ] **Streaming responses** — wrap LLM call in SSE endpoint so the UI gets token-by-token output
- [ ] **Multi-doc filtering** — add `filename` to ChromaDB `where` clause so queries can be scoped to a specific document
- [ ] **LLM integration** — add `/ask` endpoint that calls OpenAI or Anthropic with the retrieved chunks and streams the answer
- [ ] **PDF OCR fallback** — add `pytesseract` / `pdf2image` path for scanned PDFs
- [ ] **Swagger demo** — add example payloads to FastAPI route definitions for a polished `/docs` page

### Big lift (3–7 days each)

- [ ] **HuggingFace Spaces deployment** — containerise the ChromaDB server + FastAPI app as a Gradio or Streamlit Space with a hosted LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.2` via Inference API)
- [ ] **UI frontend** — build the `ui` repo: drag-and-drop upload, document list, chat interface, source highlighting
- [ ] **llm-server** — build the `llm-server` repo: Kubernetes deployment of Ollama or vLLM with a stable inference endpoint

---

## Recommended demo tier

**Quick win**: upload `test/docs/sample.pdf` via the existing `/upload` endpoint, then hit `/search/` with a question about the document, and show the returned chunks in a terminal.

**Justification**: The full ingestion-to-retrieval pipeline is already working end-to-end. A 2-minute terminal demo with `curl` or the `/docs` Swagger UI proves the core RAG mechanics — centroid routing, chunking, and vector retrieval — without any additional build work. This is the most time-efficient showcase for a portfolio review or technical interview.

**Next step up** (Medium lift): Add a `/ask` endpoint that feeds the chunks to the Anthropic or OpenAI API and returns a generated answer. This turns the demo from "here are chunks" to "here is an answer about your document", which is far more impressive to non-technical audiences.
