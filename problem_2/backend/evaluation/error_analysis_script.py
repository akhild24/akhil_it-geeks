import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import filter_candidates, rrf_fusion, hybrid_search
import backend.retrieval.basic_retrieval as br

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

test_queries = load_json('data/evaluation/test_queries_40.json')

br._ensure_bm25_state()
metadata = br._METADATA

print("--- ERROR ANALYSIS ---")
for i, q_item in enumerate(test_queries):
    query = q_item['query']
    ans_id = q_item['answer_message_id']
    
    cls = classify_query(query)
    
    results = hybrid_search(query, top_k=500)
    
    bm25_rank = -1
    dense_rank = -1
    rrf_rank = -1
    
    for r in results:
        if r['message_id'] == ans_id:
            bm25_rank = r.get('bm25_rank', -1)
            dense_rank = r.get('dense_rank', -1)
            rrf_rank = r.get('fused_rank', -1)
            break
            
    cat = "UNKNOWN"
    if rrf_rank > 0 and rrf_rank <= 5:
        cat = "SUCCESS (Top 5)"
    else:
        bm25_succ = bm25_rank > 0 and bm25_rank <= 5
        dense_succ = dense_rank > 0 and dense_rank <= 5
        
        if bm25_succ and not dense_succ:
            cat = "A (BM25 succeeds, dense fails)"
        elif not bm25_succ and dense_succ:
            cat = "B (Dense succeeds, BM25 fails)"
        elif bm25_succ and dense_succ:
            cat = "C (Both succeed, but RRF fails? Rare)"
        else:
            cat = "D (Both fail)"
            
        if cls.sender:
            cat += " | E (Sender filtering)"
        if cls.date_start:
            cat += " | F (Temporal filtering)"
            
    print(f"Q{i+1}: {query}")
    print(f"Target: {ans_id}, Type: {cls.query_type}")
    print(f"BM25 Rank: {bm25_rank}, Dense Rank: {dense_rank}, RRF Rank: {rrf_rank}")
    print(f"Category: {cat}\n")
