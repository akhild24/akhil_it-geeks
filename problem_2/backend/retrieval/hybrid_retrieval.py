import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

import backend.retrieval.basic_retrieval as br
from backend.query_classification.classifier import classify_query

def filter_by_sender(metadata_list: List[Dict], sender: str) -> List[int]:
    indices = []
    sender_lower = sender.lower()
    for i, m in enumerate(metadata_list):
        if m.get('sender', '').lower() == sender_lower:
            indices.append(i)
    return indices

def filter_by_date_range(metadata_list: List[Dict], start_date: datetime, end_date: datetime) -> List[int]:
    indices = []
    for i, m in enumerate(metadata_list):
        ts_str = m.get('timestamp')
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            ts = ts.replace(tzinfo=None)
            if start_date <= ts <= end_date:
                indices.append(i)
        except ValueError:
            pass
    return indices

def filter_candidates(metadata_list: List[Dict], sender: Optional[str] = None, date_start: Optional[datetime] = None, date_end: Optional[datetime] = None) -> List[int]:
    candidates = set(range(len(metadata_list)))
    
    if sender:
        sender_indices = set(filter_by_sender(metadata_list, sender))
        candidates = candidates.intersection(sender_indices)
        
    if date_start and date_end:
        date_indices = set(filter_by_date_range(metadata_list, date_start, date_end))
        candidates = candidates.intersection(date_indices)
        
    return sorted(list(candidates))

def rrf_fusion(bm25_results: List[Dict], dense_results: List[Dict], k: int = 60) -> List[Dict]:
    scores = {}
    items = {}
    
    for r in bm25_results:
        msg_id = r['message_id']
        rank = r['rank']
        if msg_id not in scores:
            scores[msg_id] = 0.0
            items[msg_id] = r.copy()
        scores[msg_id] += 1.0 / (k + rank)
        items[msg_id]['bm25_rank'] = rank
        
    for r in dense_results:
        msg_id = r['message_id']
        rank = r['rank']
        if msg_id not in scores:
            scores[msg_id] = 0.0
            items[msg_id] = r.copy()
        scores[msg_id] += 1.0 / (k + rank)
        items[msg_id]['dense_rank'] = rank
        
    fused = []
    for msg_id, score in scores.items():
        item = items[msg_id]
        item['fused_score'] = score
        fused.append(item)
        
    fused.sort(key=lambda x: x['fused_score'], reverse=True)
    
    for i, item in enumerate(fused):
        item['fused_rank'] = i + 1
        
    return fused

def hybrid_search(query: str, sender: Optional[str] = None, date_start: Optional[datetime] = None, date_end: Optional[datetime] = None, top_k: int = 5) -> List[Dict]:
    br._ensure_dense_state()
    br._ensure_bm25_state()
    
    # If explicit filters are not provided, try extracting them
    if not sender and not (date_start and date_end):
        classification = classify_query(query)
        sender = classification.sender
        date_start = classification.date_start
        date_end = classification.date_end

    candidate_indices = filter_candidates(br._METADATA, sender=sender, date_start=date_start, date_end=date_end)
    
    if not candidate_indices:
        return []
        
    tokenized_query = br._TOKENIZER.tokenize(query)
    all_bm25_scores = br._BM25.get_scores(tokenized_query)
    candidate_bm25_scores = np.array(all_bm25_scores)[candidate_indices]
    
    bm25_order = np.argsort(candidate_bm25_scores)[::-1]
    bm25_results = []
    for rank, local_idx in enumerate(bm25_order):
        global_idx = candidate_indices[local_idx]
        bm25_results.append({
            'message': br._METADATA[global_idx]['text'],
            'message_id': br._METADATA[global_idx]['id'],
            'rank': rank + 1,
            'raw_score': float(candidate_bm25_scores[local_idx]),
            'metadata': br._METADATA[global_idx]
        })
        
    query_vec = br._EMBEDDING_INDEXER.encode_query(query)
    q_norm = np.linalg.norm(query_vec)
    q_vec_norm = query_vec / q_norm if q_norm != 0 else query_vec
    
    candidate_embeddings = br._EMBEDDINGS[candidate_indices]
    doc_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    doc_vecs_norm = np.divide(candidate_embeddings, doc_norms, out=np.zeros_like(candidate_embeddings), where=doc_norms!=0)
    
    candidate_dense_scores = np.dot(doc_vecs_norm, q_vec_norm)
    dense_order = np.argsort(candidate_dense_scores)[::-1]
    
    dense_results = []
    for rank, local_idx in enumerate(dense_order):
        global_idx = candidate_indices[local_idx]
        dense_results.append({
            'message': br._METADATA[global_idx]['text'],
            'message_id': br._METADATA[global_idx]['id'],
            'rank': rank + 1,
            'raw_score': float(candidate_dense_scores[local_idx]),
            'metadata': br._METADATA[global_idx]
        })
        
    fused = rrf_fusion(bm25_results, dense_results)
    return fused[:top_k]
