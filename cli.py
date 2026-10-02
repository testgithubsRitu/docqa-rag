"""Command-line interface.

  python cli.py ingest data/            # build the index from files in a folder
  python cli.py ask "What is RAG?"      # ask a question against the saved index
"""
import argparse

from src.chunking import chunk_documents
from src.embedder import Embedder
from src.loader import load_folder
from src.qa import answer, format_context, get_llm
from src.vectorstore import VectorStore

STORE_DIR = "store"


def ingest(args):
    records = load_folder(args.folder)
    if not records:
        raise SystemExit(f"No .pdf/.txt/.md content found in {args.folder}")
    chunks = chunk_documents(records, args.chunk_size, args.chunk_overlap)
    store = VectorStore(Embedder())
    store.add(chunks)
    store.save(STORE_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(records)} pages into '{STORE_DIR}/'")


def ask(args):
    store = VectorStore.load(STORE_DIR, Embedder())
    hits = store.search(args.question, k=args.k)
    llm = get_llm() if not args.no_llm else None
    if llm is None and not args.no_llm:
        print("(No OPENAI_API_KEY set - showing retrieved passages instead of a generated answer.)\n")
    print(answer(args.question, hits, llm))
    if llm is not None:
        print("\nSources:\n" + format_context(hits))


def main():
    parser = argparse.ArgumentParser(description="Document Q&A with RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of documents")
    p_ingest.add_argument("folder")
    p_ingest.add_argument("--chunk-size", type=int, default=500)
    p_ingest.add_argument("--chunk-overlap", type=int, default=100)
    p_ingest.set_defaults(func=ingest)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4, help="number of chunks to retrieve")
    p_ask.add_argument("--no-llm", action="store_true", help="skip the LLM, show passages only")
    p_ask.set_defaults(func=ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
