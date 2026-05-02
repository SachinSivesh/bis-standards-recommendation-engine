"""
chunker.py
----------
Splits each BIS standard's content into overlapping word-based chunks.
Every chunk retains the parent standard's metadata.

Output format per chunk:
    {
        "standard": "IS 269: 1989",
        "title":    "ORDINARY PORTLAND CEMENT, 33 GRADE",
        "text":     "chunk text …"
    }
"""

from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:       Input text to split.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to repeat at start of next chunk.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap  # advance with overlap
    return chunks


def build_chunks(standards: List[Dict], chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    """
    Convert a list of standard dicts into a flat list of chunk dicts.

    Args:
        standards:  Output of pdf_processor.parse_standards()
        chunk_size: Words per chunk.
        overlap:    Overlap words between consecutive chunks.

    Returns:
        List of dicts: {standard, title, text}
    """
    all_chunks = []
    for item in standards:
        std   = item["standard"]
        title = item["title"]
        content = item["content"]

        # Always add at least one chunk (the full content if short)
        for chunk_text_str in chunk_text(content, chunk_size, overlap):
            all_chunks.append({
                "standard": std,
                "title":    title,
                "text":     chunk_text_str
            })

    return all_chunks


if __name__ == "__main__":
    # Quick test
    sample = [
        {
            "standard": "IS 269: 1989",
            "title": "ORDINARY PORTLAND CEMENT, 33 GRADE",
            "content": "IS 269 : 1989 ORDINARY PORTLAND CEMENT " + "word " * 600
        }
    ]
    chunks = build_chunks(sample)
    print(f"Generated {len(chunks)} chunks for the sample standard.")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c['text'].split())} words | std={c['standard']}")
