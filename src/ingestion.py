import os
from pypdf import PdfReader
from typing import List

def load_pdf(path: str) -> str:
    """Load a PDF and return all text as a single string."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_txt(path: str) -> str:
    """Load a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_document(path: str) -> str:
    """Auto-detect file type and load."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = 80, overlap: int = 40) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    
    chunk_size = words per chunk
    overlap    = words shared between consecutive chunks
    
    Why overlap? So a sentence that falls on a boundary
    doesn't get cut in half and lose context.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # slide forward with overlap

    return [c for c in chunks if len(c) > 30]  # drop tiny chunks

def load_and_chunk(path: str, chunk_size: int = 80, overlap: int = 40) -> List[str]:
    """One-call convenience: load file and return chunks."""
    text = load_document(path)
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Loaded: {path}")
    print(f"Total characters: {len(text)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) // len(chunks)} chars")
    return chunks
