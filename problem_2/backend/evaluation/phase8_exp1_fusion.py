import json
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import filter_candidates
import backend.retrieval.basic_retrieval as br

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

test_queries = load_json('data/evaluation/test_queries_40.json')
br._ensure_bm25_state()
br._ensure_dense_state()
metadata = br._METADATA

def minmax_normalize(scores):
    if len(scores) == 0:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def score_fusion(bm25_results, dense_results, alpha=0.3):
    # alpha is weight for BM25. (1-alpha) is weight for Dense.
    scores = {}
    items = {}
    
    # Normalize BM25
    b_scores = [r['raw_score'] for r in bm25_results]
    b_norm = minmax_normalize(b_scores)
    for i, r in enumerate(bm25_results):
        msg_id = r['message_id']
        if msg_id not in items:
            items[msg_id] = r.copy()
            scores[msg_id] = {'b': 0.0, 'd': 0.0}
        scores[msg_id]['b'] = b_norm[i]
        items[msg_id]['bm25_rank'] = r['rank']
        
    # Normalize Dense
    d_scores = [r['raw_score'] for r in dense_results]
    d_norm = minmax_normalize(d_scores)
    for i, r in enumerate(dense_results):
        msg_id = r['message_id']
        if msg_id not in items:
            items[msg_id] = r.copy()
            scores[msg_id] = {'b': 0.0, 'd': 0.0}
        scores[msg_id]['d'] = d_norm[i]
        items[msg_id]['dense_rank'] = r['rank']
        
    fused = []
    for msg_id, s in scores.items():
        item = items[msg_id]
        # Convex combination
        # Or alternative: max(s['b'], s['d'])
        item['fused_score'] = alpha * s['b'] + (1 - alpha) * s['d']
        fused.append(item)
        
    fused.sort(key=lambda x: x['fused_score'], reverse=True)
    for i, item in enumerate(fused):
        item['fused_rank'] = i + 1
        
    return fused

def evaluate_fusion(alpha):
    success_count = 0
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    for q_item in test_queries:
        query = q_item['query']
        ans_id = q_item['answer_message_id']
        
        cls = classify_query(query)
        candidates = filter_candidates(metadata, cls.sender, cls.date_start, cls.date_end)
        
        if not candidates:
            ranks.append(0)
            continue
            
        tokenized_query = br._TOKENIZER.tokenize(query)
        all_bm25_scores = br._BM25.get_scores(tokenized_query)
        candidate_bm25_scores = np.array(all_bm25_scores)[candidates]
        bm25_order = np.argsort(candidate_bm25_scores)[::-1]
        
        bm25_results = []
        for rank, local_idx in enumerate(bm25_order):
            global_idx = candidates[local_idx]
            bm25_results.append({
                'message': metadata[global_idx]['text'],
                'message_id': metadata[global_idx]['id'],
                'rank': rank + 1,
                'raw_score': float(candidate_bm25_scores[local_idx]),
            })
            
        query_vec = br._EMBEDDING_INDEXER.encode_query(query)
        q_norm = np.linalg.norm(query_vec)
        q_vec_norm = query_vec / q_norm if q_norm != 0 else query_vec
        candidate_embeddings = br._EMBEDDINGS[candidates]
        doc_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        doc_vecs_norm = np.divide(candidate_embeddings, doc_norms, out=np.zeros_like(candidate_embeddings), where=doc_norms!=0)
        
        candidate_dense_scores = np.dot(doc_vecs_norm, q_vec_norm)
        dense_order = np.argsort(candidate_dense_scores)[::-1]
        
        dense_results = []
        for rank, local_idx in enumerate(dense_order):
            global_idx = candidates[local_idx]
            dense_results.append({
                'message': metadata[global_idx]['text'],
                'message_id': metadata[global_idx]['id'],
                'rank': rank + 1,
                'raw_score': float(candidate_dense_scores[local_idx]),
            })
            
        results = score_fusion(bm25_results, dense_results, alpha=alpha)
        
        rank = -1
        for i, r in enumerate(results):
            if r['message_id'] == ans_id:
                rank = i + 1
                break
                
        if rank != -1:
            ranks.append(rank)
            success_count += 1
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
        else:
            ranks.append(0)

    total = len(test_queries)
    mrr = sum([1/r for r in ranks if r > 0]) / total
    
    print(f"Alpha {alpha:.1f} (BM25 weight): Hit@1={hits_at_1}, Hit@5={hits_at_5}, MRR={mrr:.4f}")
    return hits_at_1, hits_at_5, mrr

print("--- EXPERIMENT 1: SCORE NORMALIZATION FUSION ---")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    evaluate_fusion(alpha)
