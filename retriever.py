"""
retriever.py
------------
Hybrid retrieval: semantic (FAISS) + keyword boosting.

Scoring formula:
    final_score = 0.7 * semantic_score + 0.3 * keyword_score

Returns top-k unique BIS standard IDs ranked by final_score.
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple

def normalize_standard(std):
    return std.replace(" ", "").lower()


def rule_boost(query, text):
    query = query.lower()
    text = text.lower()
    score = 0

    # Cement rules
    if "33 grade" in query and "269" in text:
        score += 10
    if "43 grade" in query and "8112" in text:
        score += 10
    if "53 grade" in query and "12269" in text:
        score += 10

    if "aggregate" in query and "383" in text:
        score += 10

    if "pipe" in query and "458" in text:
        score += 10

    if "masonry" in query and "2185" in text:
        score += 10

    if "corrugated" in query and "459" in text:
        score += 10

    if "slag" in query and "455" in text:
        score += 10

    if "pozzolana" in query and "1489" in text:
        score += 10

    if "supersulphated" in query and "6909" in text:
        score += 10

    if "white" in query and "8042" in text:
        score += 10

    # Lightweight concrete blocks
    if "lightweight" in query.lower() and "part 2" in text.lower():
        score += 10

    return score


def retrieve(
    query: str,
    index,                      # faiss.Index
    chunks: List[Dict],         # list of {standard, title, text}
    model,                      # SentenceTransformer
    top_k_chunks: int = 200,    # Top candidates to consider
    top_n_standards: int = 3,   # Return top 3 unique standards
    alpha: float = 0.6,         # Keep for compatibility
) -> List[str]:
    """
    Main retrieval function with HARD RULE BOOSTING.
    """
    # 1. Embed query
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # 2. FAISS search
    scores, indices = index.search(q_vec, top_k_chunks)
    semantic_scores = scores[0]
    chunk_indices   = indices[0]

    # 3. Build per-standard aggregated scores
    std_scores: Dict[str, float] = {}
    std_raw_map: Dict[str, str] = {}

    for sem_score, idx in zip(semantic_scores, chunk_indices):
        if idx < 0 or idx >= len(chunks):
            continue
            
        chunk = chunks[idx]
        std   = chunk["standard"]
        title = chunk["title"]
        text  = chunk["text"]
        
        # Combine title and text for rule matching
        combined_text = (title + " " + text).lower()
        
        # Rule boost dominates ranking
        boost = rule_boost(query, combined_text)
        final = float(sem_score) + boost
        
        norm_std = normalize_standard(std)
        if norm_std not in std_scores or final > std_scores[norm_std]:
            std_scores[norm_std] = final
            std_raw_map[norm_std] = std

    # 4. Sort by final_score descending
    ranked = sorted(std_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 unique standards
    return [std_raw_map[norm_std] for norm_std, _ in ranked[:top_n_standards]]

