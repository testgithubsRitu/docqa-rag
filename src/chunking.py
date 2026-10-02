"""Split documents into overlapping chunks.

Why overlap? A sentence that straddles a chunk boundary would otherwise be cut in half and
become hard to retrieve. A small overlap keeps that context in both neighbouring chunks.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(records, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for record in records:
        for piece in splitter.split_text(record["text"]):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": piece.strip(),
                    "source": record["source"],
                    "page": record["page"],
                }
            )
    return [c for c in chunks if c["text"]]
