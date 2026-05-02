"""
build_index.py
--------------
One-time script that:
  1. Extracts BIS standards from data/dataset.pdf
  2. Chunks each standard's content
  3. Embeds all chunks with sentence-transformers
  4. Builds & saves a FAISS index

Run once before inference:
    python build_index.py
"""

from pdf_processor import load_or_extract
from chunker import build_chunks
from embedder import build_and_save


def main():
    # Step 1: Load / extract standards from PDF
    standards = load_or_extract("data/dataset.pdf", cache_path="data/standards_cache.json")
    print(f"\n[Build Index] {len(standards)} standards loaded.")

    # Step 2: Chunk
    chunks = build_chunks(standards, chunk_size=400, overlap=80)
    print(f"[Build Index] {len(chunks)} chunks created.")

    # Step 3: Embed + index + save
    build_and_save(chunks)
    print("[Build Index] Index build complete. Ready for inference.")


if __name__ == "__main__":
    main()
