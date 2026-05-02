import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import sys

# Add e:\BIS to path
sys.path.append(r"e:\BIS")

from retriever import retrieve, keyword_score, intent_score, normalize_standard
from embedder import load_artefacts, get_model

def debug_query(query_id, query_text, expected):
    index, chunks, embeddings = load_artefacts()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    q_vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, 50)
    
    print(f"\nQuery: {query_id} - {query_text}")
    print(f"Expected: {expected}")
    
    found_expected = False
    norm_expected = normalize_standard(expected)
    
    # Track best scores per standard
    std_best = {}
    
    for i, (idx, sem_score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0: continue
        chunk = chunks[idx]
        std = chunk["standard"]
        text = chunk["text"]
        norm_std = normalize_standard(std)
        
        kw = keyword_score(query_text, text)
        intent = intent_score(query_text, text)
        final = 0.6 * float(sem_score) + 0.2 * kw + 0.2 * intent
        
        if norm_std not in std_best or final > std_best[norm_std][0]:
            std_best[norm_std] = (final, sem_score, kw, intent, std)
            
    # Sort standards by best final score
    ranked_stds = sorted(std_best.items(), key=lambda x: x[1][0], reverse=True)
    
    for rank, (norm_std, data) in enumerate(ranked_stds[:10]):
        final, sem, kw, intent, raw_std = data
        marker = "[MATCH]" if norm_std == norm_expected else f"[Rank {rank}]"
        print(f"  {marker} {raw_std} | Final: {final:.4f} | Sem: {sem:.4f} | KW: {kw} | Intent: {intent}")
        if norm_std == norm_expected:
            found_expected = True
            
    if not found_expected:
        print("  [ERROR] Expected standard not found in top 10 standards from top 50 chunks!")

if __name__ == "__main__":
    os.chdir(r"e:\BIS")
    debug_query("PUB-01", "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?", "IS 269: 1989")
    debug_query("PUB-08", "Which standard applies to masonry cement used for general purposes where mortars for masonry are required, but not intended for structural concrete?", "IS 3466: 1988")
