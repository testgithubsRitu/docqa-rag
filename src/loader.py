"""Load PDF / text files into a simple list of page-level records."""
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_document(path):
    """Return a list of {"source", "page", "text"} records for one file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(p))
        records = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append({"source": p.name, "page": number, "text": text})
        return records
    if suffix in {".txt", ".md"}:
        return [{"source": p.name, "page": 1, "text": p.read_text(encoding="utf-8")}]
    raise ValueError(f"Unsupported file type: {p.suffix}")


def load_folder(folder):
    """Load every supported file in a folder (non-recursive)."""
    records = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in SUPPORTED:
            records.extend(load_document(p))
    return records
