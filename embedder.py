"""
embedder.py
-----------
Converts text chunks into dense embedding vectors using
sentence-transformers (all-MiniLM-L6-v2) and builds a FAISS
flat-L2 index for nearest-neighbour retrieval.

Saved artefacts (in data/):
    embeddings.npy   – numpy float32 array, shape [N, 384]
    faiss.index      – FAISS index file
    chunks_meta.json – list of {standard, title, text} dicts (N entries)
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

MODEL_NAME  = "all-MiniLM-L6-v2"
INDEX_PATH  = "data/faiss.index"
META_PATH   = "data/chunks_meta.json"
EMB_PATH    = "data/embeddings.npy"


def get_model() -> SentenceTransformer:
    print(f"[Embedder] Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64
) -> np.ndarray:
    """
    Encode the 'text' field of each chunk dict into a float32 embedding.
    Returns shape [N, embedding_dim].
    """
    texts = [c["text"] for c in chunks]
    print(f"[Embedder] Encoding {len(texts)} chunks (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # unit vectors → cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an inner-product (cosine) FAISS index.
    (Embeddings are already L2-normalised, so IP == cosine similarity.)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[Embedder] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_artefacts(
    index: faiss.Index,
    chunks: List[Dict],
    embeddings: np.ndarray
) -> None:
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embeddings)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[Embedder] Saved index -> {INDEX_PATH}")
    print(f"[Embedder] Saved meta  -> {META_PATH}")
    print(f"[Embedder] Saved embs  -> {EMB_PATH}")


def load_artefacts():
    """
    Returns (index, chunks, embeddings) from disk.
    Raises FileNotFoundError if any artefact is missing.
    """
    if not all(os.path.exists(p) for p in [INDEX_PATH, META_PATH, EMB_PATH]):
        raise FileNotFoundError(
            "FAISS artefacts not found. Run build_index.py first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMB_PATH)
    print(f"[Embedder] Loaded index with {index.ntotal} vectors.")
    return index, chunks, embeddings


def build_and_save(chunks: List[Dict]) -> None:
    """Full pipeline: embed → index → save."""
    model      = get_model()
    embeddings = embed_chunks(chunks, model)
    index      = build_faiss_index(embeddings)
    save_artefacts(index, chunks, embeddings)
