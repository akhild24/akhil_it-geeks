import numpy as np
import os
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

def filter_by_date_range(
    metadata_list: List[Dict],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[int]:
    """Return messages inside optional inclusive date bounds."""
    if start_date and start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    if end_date and end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
    indices = []
    for i, m in enumerate(metadata_list):
        ts_str = m.get('timestamp')
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            ts = ts.replace(tzinfo=None)
            if (start_date is None or start_date <= ts) and (end_date is None or ts <= end_date):
                indices.append(i)
        except ValueError:
            pass
    return indices

def filter_candidates(metadata_list: List[Dict], sender: Optional[str] = None, date_start: Optional[datetime] = None, date_end: Optional[datetime] = None) -> List[int]:
    candidates = set(range(len(metadata_list)))
    
    if sender:
        sender_indices = set(filter_by_sender(metadata_list, sender))
        candidates = candidates.intersection(sender_indices)
        
    if date_start or date_end:
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

def minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def score_fusion(bm25_results: List[Dict], dense_results: List[Dict], alpha: float = 0.2) -> List[Dict]:
    scores = {}
    items = {}
    
    b_scores = [r['raw_score'] for r in bm25_results]
    b_norm = minmax_normalize(b_scores)
    
    for i, r in enumerate(bm25_results):
        msg_id = r['message_id']
        if msg_id not in items:
            items[msg_id] = r.copy()
            scores[msg_id] = {'b': 0.0, 'd': 0.0}
        scores[msg_id]['b'] = b_norm[i]
        items[msg_id]['bm25_rank'] = r['rank']
        
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
        item['fused_score'] = alpha * s['b'] + (1.0 - alpha) * s['d']
        fused.append(item)
        
    fused.sort(key=lambda x: x['fused_score'], reverse=True)
    
    for i, item in enumerate(fused):
        item['fused_rank'] = i + 1
        
    return fused

def hybrid_search(query: str, sender: Optional[str] = None, date_start: Optional[datetime] = None, date_end: Optional[datetime] = None, top_k: int = 5) -> List[Dict]:
    br._ensure_dense_state()
    br._ensure_bm25_state()
    
    # Explicit values take precedence, while omitted dimensions can still be
    # inferred from the natural-language query.
    classification = classify_query(query)
    sender = sender or classification.sender
    date_start = date_start or classification.date_start
    date_end = date_end or classification.date_end

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
        
    # RRF is designed to fuse the two retrieval heads' strongest candidates.
    # Including every message (especially the BM25 zero-score tail) gives
    # arbitrary long-tail ranks a vote and degrades relevance.
    fusion_depth = min(100, len(candidate_indices))
    fused = rrf_fusion(bm25_results[:fusion_depth], dense_results[:fusion_depth])
        
    # Phase 9: Confidence-Aware Retrieval (No-match detection)
    # Strategy: top5_overlap_dense @ threshold 0.518
    # Formula: top_dense_raw + (bm25_dense_top5_overlap * 0.05)
    
    top_dense_raw = dense_results[0]['raw_score'] if dense_results else 0.0
    top_bm25_raw = bm25_results[0]['raw_score'] if bm25_results else 0.0
    bm25_top5_ids = set(r['message_id'] for r in bm25_results[:5])
    dense_top5_ids = set(r['message_id'] for r in dense_results[:5])
    top5_overlap = len(bm25_top5_ids & dense_top5_ids)
    
    confidence_score = top_dense_raw + (top5_overlap * 0.05)
    stopwords = {
        'a', 'about', 'an', 'and', 'are', 'can', 'did', 'do', 'everyone',
        'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'out', 'the',
        'they', 'to', 'was', 'we', 'what', 'when', 'where', 'which', 'who',
    }
    query_terms = set(token for token in tokenized_query if token not in stopwords)
    lexical_coverage = 0.0
    if query_terms:
        for result in bm25_results[:100]:
            indexed_terms = br._TOKENIZER.tokenize(
                result['message'] + ' ' + result['metadata'].get('search_context', '')
            )
            lexical_coverage = max(
                lexical_coverage,
                len(query_terms.intersection(indexed_terms)) / len(query_terms),
            )
    
    # Optional bypass for baseline testing
    if os.environ.get("DISABLE_CONFIDENCE_THRESHOLD") != "1":
        # A low dense score alone is not enough to reject a valid query: short
        # Hinglish messages can still have partial lexical evidence. Reject
        # low-confidence candidates when the lexical evidence covers too little
        # of the meaningful query, which filters generic-word false matches.
        if confidence_score < 0.518 and lexical_coverage < 0.5:
            return []
            
    return fused[:top_k]
