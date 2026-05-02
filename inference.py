"""
inference.py
------------
BIS Standards Recommendation Engine – RAG Inference

Usage:
    python inference.py --input data/public_test_set.json --output output.json

Output format (matches sample_output.json exactly):
[
  {
    "id": "PUB-01",
    "query": "...",
    "expected_standards": [...],
    "retrieved_standards": [...],
    "latency_seconds": 1.24
  },
  ...
]
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Lazy imports (so build errors surface clearly)
# ---------------------------------------------------------------------------
def _import_dependencies():
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

_import_dependencies()

from pdf_processor import load_or_extract
from chunker       import build_chunks
from embedder      import build_and_save, load_artefacts, get_model, INDEX_PATH, META_PATH, EMB_PATH
from retriever     import retrieve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_PATH   = "data/dataset.pdf"
CACHE_PATH = "data/standards_cache.json"


def ensure_index() -> tuple:
    """
    Return (index, chunks, model).
    Builds the index from scratch if artefacts are missing.
    """
    model = get_model()

    if not all(os.path.exists(p) for p in [INDEX_PATH, META_PATH, EMB_PATH]):
        print("[Inference] FAISS index not found – building now …")
        standards = load_or_extract(PDF_PATH, cache_path=CACHE_PATH)
        chunks    = build_chunks(standards, chunk_size=400, overlap=80)
        build_and_save(chunks)

    index, chunks, _ = load_artefacts()
    return index, chunks, model


def run_inference(input_path: str, output_path: str) -> None:
    """
    Read queries from input_path, run retrieval for each, write output_path.
    """
    # ---- Load input --------------------------------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        print("[ERROR] Input JSON must be a list of query objects.")
        sys.exit(1)

    print(f"[Inference] Loaded {len(queries)} queries from {input_path}")

    # ---- Ensure index is ready --------------------------------------------
    index, chunks, model = ensure_index()

    # ---- Run retrieval for each query -------------------------------------
    results = []
    for item in queries:
        query_id = item.get("id", "")
        query    = item.get("query", "")
        expected = item.get("expected_standards", [])

        t0 = time.perf_counter()
        retrieved = retrieve(
            query=query,
            index=index,
            chunks=chunks,
            model=model,
            top_k_chunks=200,
            top_n_standards=5,
            alpha=0.6
        )
        latency = round(time.perf_counter() - t0, 4)

        result = {
            "id":                  query_id,
            "retrieved_standards": retrieved,
            "latency_seconds":     latency
        }
        results.append(result)
        print(f"  [{query_id}] latency={latency:.3f}s  top={retrieved[:3]}")

    # ---- Write output ------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Inference] Results written to {output_path}")
    print(f"[Inference] Done. {len(results)} queries processed.")


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine – RAG Inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (list of {id, query, expected_standards})"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON results"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
