# DocQA-RAG: Chat with Your Documents

A small, readable **Retrieval-Augmented Generation (RAG)** pipeline in Python. Point it at a folder of
PDFs / text files, then ask questions in plain English and get the most relevant passages (or an
LLM-written answer) with **source and page references**.

Built to understand how RAG works end to end: chunking, embeddings, vector search and grounded prompting.

## How it works

```
 documents ──► load ──► chunk (overlapping) ──► embed (MiniLM) ──► FAISS index
                                                                      │
 question ──► embed ──► nearest-neighbour search ──► top-k chunks ────┤
                                                                      ▼
                                   extractive answer  OR  LangChain prompt | LLM | parser
```

|Stage|File|What it does|
|-|-|-|
|Load|`src/loader.py`|Reads PDF (page by page) and `.txt`/`.md` files with `pypdf`|
|Chunk|`src/chunking.py`|`RecursiveCharacterTextSplitter` from LangChain, 500 chars with 100 overlap|
|Embed|`src/embedder.py`|`all-MiniLM-L6-v2` (sentence-transformers), 384-dim, L2-normalised|
|Store/search|`src/vectorstore.py`|FAISS `IndexFlatIP`; on normalised vectors, inner product = cosine similarity|
|Answer|`src/qa.py`|Extractive mode (no key needed) or a LangChain chain with a grounded prompt|

## Quick start

```bash
git clone <https://github.com/testgithubsRitu/docqa-rag> \&\& cd docqa-rag
python -m venv .venv \&\& source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python cli.py ingest data/                              # builds ./store (index + chunk metadata)
python cli.py ask "Why does RAG reduce hallucination?"
```

Without an API key you get the top matching passages with similarity scores. To get a written answer,
set `OPENAI\_API\_KEY` (optionally `OPENAI\_MODEL`) and run the same `ask` command. Use `--no-llm` to force
passage-only output. The LangChain chain in `src/qa.py` can be pointed at any chat model.

Put your own PDFs in `data/` and re-run `ingest` to query them.

## Tests

```bash
pytest -q
```

Tests use a small deterministic embedder, so they run offline in about a second. They cover chunk size and
metadata, correct top-1 retrieval, and saving/loading the index.

## Design choices and limitations

* **Chunk size / overlap** are CLI flags (`--chunk-size`, `--chunk-overlap`); retrieval quality is sensitive to both.
* **Exact search** (`IndexFlatIP`) is fine for thousands of chunks; for millions, switch to an approximate FAISS index (IVF/HNSW).
* Scanned PDFs without a text layer need OCR first (not included).
* Answers are only as good as the retrieved context; the prompt tells the model to say "I don't know" otherwise.

## Ideas for next steps

* Swap FAISS for ChromaDB or Pinecone behind the same `VectorStore` interface
* Add a reranker (cross-encoder) after retrieval
* Measure retrieval quality (hit-rate@k / MRR) on a small labelled question set
* Streamlit UI

