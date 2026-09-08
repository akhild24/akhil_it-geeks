"""
Recall AI — Phase 9A/9B Confidence Analysis

Runs all 50 evaluation queries (40 valid + 10 no-match) through the CURRENT
Phase 8 retrieval pipeline, collects comprehensive confidence signals,
and performs statistical analysis to determine whether valid and no-match
queries can be reliably separated.

DOES NOT MODIFY production code. Creates experiment artifacts only.
"""

import json
import os
import sys
import time
import hashlib
import numpy as np
from datetime import datetime
from statistics import mean, median, stdev

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import hybrid_search, filter_candidates, score_fusion, minmax_normalize
import backend.retrieval.basic_retrieval as br


# ========== STEP 0: FROZEN ARTIFACT INTEGRITY CHECK ==========

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def check_frozen_artifacts():
    frozen_files = {
        'data/chat_corpus_2.json': 'E2EA4FB9',
        'data/evaluation/test_queries_40.json': '6D02FF78',
        'data/evaluation/no_match_queries.json': 'A1AB3A2B',
        'data/indexes/corpus_2_embeddings.npy': 'C9869EE8',
        'data/indexes/corpus_2_bm25.pkl': 'BBE2BD6B',
        'data/indexes/corpus_2_metadata.json': 'D6DE404A',
    }
    print("=" * 60)
    print("FROZEN ARTIFACT INTEGRITY CHECK")
    print("=" * 60)
    all_pass = True
    hashes = {}
    for fpath, expected_prefix in frozen_files.items():
        full_path = os.path.join(os.path.dirname(__file__), '..', '..', fpath)
        if not os.path.exists(full_path):
            print(f"  [MISSING] {fpath}")
            all_pass = False
            continue
        h = sha256_file(full_path)
        hashes[fpath] = h
        if h.startswith(expected_prefix):
            print(f"  [PASS] {fpath}: {h[:16]}...")
        else:
            print(f"  [FAIL] {fpath}: {h[:16]}... (expected prefix {expected_prefix})")
            all_pass = False
    
    if all_pass:
        print("  All frozen artifacts verified.\n")
    else:
        print("  WARNING: Frozen artifact mismatch detected!\n")
    return all_pass, hashes


# ========== STEP 1: LOAD DATA ==========

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========== STEP 2: COLLECT CONFIDENCE DATA ==========

def collect_confidence_data(query_text, expected_id, is_no_match, top_k_collect=100):
    """
    Run a single query through the FULL Phase 8 retrieval pipeline and
    collect comprehensive confidence signals.
    
    Uses top_k=100 to get enough candidates for analysis, but also collects
    raw BM25/dense scores independently for signal analysis.
    """
    # Classify query
    cls = classify_query(query_text)
    
    # Get candidate indices (same as hybrid_search does)
    candidate_indices = filter_candidates(br._METADATA, sender=cls.sender,
                                          date_start=cls.date_start, date_end=cls.date_end)
    candidate_count = len(candidate_indices)
    
    if candidate_count == 0:
        return {
            'query': query_text,
            'expected_id': expected_id,
            'is_no_match': is_no_match,
            'query_type': cls.query_type,
            'candidate_count': 0,
            'top_result_id': None,
            'top_fused_score': 0.0,
            'second_fused_score': 0.0,
            'fused_margin': 0.0,
            'top_bm25_raw': 0.0,
            'top_dense_raw': 0.0,
            'top_bm25_rank': None,
            'top_dense_rank': None,
            'top_fused_rank': None,
            'bm25_dense_agreement': False,
            'in_top_1': False,
            'in_top_5': False,
            'in_top_10': False,
            'expected_fused_rank': None,
            # Raw score arrays for deep analysis
            'all_dense_scores_top10': [],
            'all_bm25_scores_top10': [],
            'all_fused_scores_top10': [],
        }
    
    # ---- BM25 retrieval (full candidate set) ----
    tokenized_query = br._TOKENIZER.tokenize(query_text)
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
    
    # ---- Dense retrieval (full candidate set) ----
    query_vec = br._EMBEDDING_INDEXER.encode_query(query_text)
    q_norm = np.linalg.norm(query_vec)
    q_vec_norm = query_vec / q_norm if q_norm != 0 else query_vec
    
    candidate_embeddings = br._EMBEDDINGS[candidate_indices]
    doc_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    doc_vecs_norm = np.divide(candidate_embeddings, doc_norms,
                              out=np.zeros_like(candidate_embeddings), where=doc_norms != 0)
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
    
    # ---- Phase 8 score fusion (alpha=0.2) ----
    fused_results = score_fusion(bm25_results, dense_results, alpha=0.2)
    
    # Collect top result info
    top_result = fused_results[0] if fused_results else None
    second_result = fused_results[1] if len(fused_results) > 1 else None
    
    top_fused_score = top_result['fused_score'] if top_result else 0.0
    second_fused_score = second_result['fused_score'] if second_result else 0.0
    fused_margin = top_fused_score - second_fused_score
    
    top_result_id = top_result['message_id'] if top_result else None
    
    # Get BM25/dense ranks for the top fused result
    top_bm25_rank = top_result.get('bm25_rank') if top_result else None
    top_dense_rank = top_result.get('dense_rank') if top_result else None
    
    # Raw top BM25 and Dense scores (absolute top-1 from each retriever)
    top_bm25_raw = bm25_results[0]['raw_score'] if bm25_results else 0.0
    top_dense_raw = dense_results[0]['raw_score'] if dense_results else 0.0
    
    # BM25/Dense agreement: does the top BM25 result match the top Dense result?
    bm25_top_id = bm25_results[0]['message_id'] if bm25_results else None
    dense_top_id = dense_results[0]['message_id'] if dense_results else None
    bm25_dense_agreement = (bm25_top_id == dense_top_id)
    
    # BM25/Dense top-5 overlap
    bm25_top5_ids = set(r['message_id'] for r in bm25_results[:5])
    dense_top5_ids = set(r['message_id'] for r in dense_results[:5])
    bm25_dense_top5_overlap = len(bm25_top5_ids & dense_top5_ids)
    
    # Check expected answer rank
    expected_fused_rank = None
    in_top_1 = False
    in_top_5 = False
    in_top_10 = False
    
    if expected_id is not None:
        for r in fused_results:
            if r['message_id'] == expected_id:
                expected_fused_rank = r['fused_rank']
                if expected_fused_rank == 1:
                    in_top_1 = True
                if expected_fused_rank <= 5:
                    in_top_5 = True
                if expected_fused_rank <= 10:
                    in_top_10 = True
                break
    
    # Collect score arrays for top-10 fused results
    all_fused_top10 = [r['fused_score'] for r in fused_results[:10]]
    all_dense_top10 = [dense_results[i]['raw_score'] for i in range(min(10, len(dense_results)))]
    all_bm25_top10 = [bm25_results[i]['raw_score'] for i in range(min(10, len(bm25_results)))]
    
    # Additional signals: raw dense score of top fused result (not top dense result)
    # This tells us how semantically relevant the best fused match actually is
    top_fused_dense_raw = None
    top_fused_bm25_raw = None
    if top_result:
        top_msg_id = top_result['message_id']
        for r in dense_results:
            if r['message_id'] == top_msg_id:
                top_fused_dense_raw = r['raw_score']
                break
        for r in bm25_results:
            if r['message_id'] == top_msg_id:
                top_fused_bm25_raw = r['raw_score']
                break
    
    return {
        'query': query_text,
        'expected_id': expected_id,
        'is_no_match': is_no_match,
        'query_type': cls.query_type,
        'candidate_count': candidate_count,
        'top_result_id': top_result_id,
        'top_fused_score': round(top_fused_score, 6),
        'second_fused_score': round(second_fused_score, 6),
        'fused_margin': round(fused_margin, 6),
        'top_bm25_raw': round(top_bm25_raw, 6),
        'top_dense_raw': round(top_dense_raw, 6),
        'top_fused_dense_raw': round(top_fused_dense_raw, 6) if top_fused_dense_raw is not None else None,
        'top_fused_bm25_raw': round(top_fused_bm25_raw, 6) if top_fused_bm25_raw is not None else None,
        'top_bm25_rank': top_bm25_rank,
        'top_dense_rank': top_dense_rank,
        'top_fused_rank': 1 if top_result else None,
        'bm25_dense_agreement': bm25_dense_agreement,
        'bm25_dense_top5_overlap': bm25_dense_top5_overlap,
        'in_top_1': in_top_1,
        'in_top_5': in_top_5,
        'in_top_10': in_top_10,
        'expected_fused_rank': expected_fused_rank,
        'all_fused_scores_top10': [round(s, 6) for s in all_fused_top10],
        'all_dense_scores_top10': [round(s, 6) for s in all_dense_top10],
        'all_bm25_scores_top10': [round(s, 6) for s in all_bm25_top10],
    }


# ========== STEP 3: STATISTICAL ANALYSIS ==========

def compute_stats(values, label=""):
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None, 'std': None,
                'q25': None, 'q75': None}
    arr = np.array(values)
    result = {
        'count': len(values),
        'min': round(float(np.min(arr)), 6),
        'max': round(float(np.max(arr)), 6),
        'mean': round(float(np.mean(arr)), 6),
        'median': round(float(np.median(arr)), 6),
        'std': round(float(np.std(arr)), 6),
        'q25': round(float(np.percentile(arr, 25)), 6),
        'q75': round(float(np.percentile(arr, 75)), 6),
    }
    return result


def analyze_signal_separation(valid_values, no_match_values, signal_name):
    """Analyze whether a signal separates valid from no-match queries."""
    v_stats = compute_stats(valid_values, f"valid_{signal_name}")
    nm_stats = compute_stats(no_match_values, f"nomatch_{signal_name}")
    
    # Compute overlap
    if v_stats['min'] is not None and nm_stats['max'] is not None:
        overlap_lower = max(v_stats['min'], nm_stats['min'])
        overlap_upper = min(v_stats['max'], nm_stats['max'])
        has_overlap = overlap_lower <= overlap_upper
    else:
        has_overlap = True
    
    return {
        'signal': signal_name,
        'valid': v_stats,
        'no_match': nm_stats,
        'overlap_exists': has_overlap,
    }


# ========== STEP 4: THRESHOLD EXPERIMENTS ==========

def test_threshold(valid_data, no_match_data, signal_key, threshold, reject_below=True):
    """
    Test a threshold on a given signal.
    reject_below=True: reject queries with signal < threshold
    reject_below=False: reject queries with signal > threshold
    """
    # No-match rejection
    nm_rejected = 0
    nm_total = len(no_match_data)
    for d in no_match_data:
        val = d.get(signal_key, 0.0)
        if val is None:
            val = 0.0
        if reject_below and val < threshold:
            nm_rejected += 1
        elif not reject_below and val > threshold:
            nm_rejected += 1
    
    # Valid query rejection (false negatives)
    v_rejected = 0
    v_total = len(valid_data)
    v_hits_1 = 0
    v_hits_5 = 0
    v_mrr_sum = 0.0
    
    for d in valid_data:
        val = d.get(signal_key, 0.0)
        if val is None:
            val = 0.0
        rejected = False
        if reject_below and val < threshold:
            rejected = True
        elif not reject_below and val > threshold:
            rejected = True
        
        if rejected:
            v_rejected += 1
            # Rejected query = no result returned
            continue
        
        # Not rejected - count hits
        if d['in_top_1']:
            v_hits_1 += 1
        if d['in_top_5']:
            v_hits_5 += 1
        if d['expected_fused_rank'] and d['expected_fused_rank'] > 0:
            v_mrr_sum += 1.0 / d['expected_fused_rank']
    
    nm_rejection_rate = nm_rejected / nm_total if nm_total > 0 else 0
    v_rejection_rate = v_rejected / v_total if v_total > 0 else 0
    fpr = (nm_total - nm_rejected) / nm_total if nm_total > 0 else 0  # no-match that got through
    fnr = v_rejected / v_total if v_total > 0 else 0
    
    hit_1 = v_hits_1 / v_total if v_total > 0 else 0
    hit_5 = v_hits_5 / v_total if v_total > 0 else 0
    mrr = v_mrr_sum / v_total if v_total > 0 else 0
    
    return {
        'signal': signal_key,
        'threshold': round(threshold, 6),
        'reject_below': reject_below,
        'nm_rejected': nm_rejected,
        'nm_total': nm_total,
        'nm_rejection_rate': round(nm_rejection_rate, 4),
        'v_rejected': v_rejected,
        'v_total': v_total,
        'v_rejection_rate': round(v_rejection_rate, 4),
        'fpr': round(fpr, 4),  # false positive rate (no-match queries returning results)
        'fnr': round(fnr, 4),  # false negative rate (valid queries rejected)
        'hit_at_1': round(hit_1, 4),
        'hit_at_5': round(hit_5, 4),
        'mrr': round(mrr, 4),
    }


# ========== STEP 5: COMBINED SIGNAL EXPERIMENTS ==========

def compute_combined_score(data_item, formula_name):
    """Compute combined confidence scores using different deterministic formulas."""
    top_dense = data_item.get('top_dense_raw', 0.0) or 0.0
    top_bm25 = data_item.get('top_bm25_raw', 0.0) or 0.0
    margin = data_item.get('fused_margin', 0.0) or 0.0
    fused = data_item.get('top_fused_score', 0.0) or 0.0
    agreement = 1.0 if data_item.get('bm25_dense_agreement', False) else 0.0
    top5_overlap = data_item.get('bm25_dense_top5_overlap', 0) or 0
    top_fused_dense = data_item.get('top_fused_dense_raw', 0.0) or 0.0
    top_fused_bm25 = data_item.get('top_fused_bm25_raw', 0.0) or 0.0
    
    if formula_name == 'dense_plus_margin':
        return top_dense + margin
    elif formula_name == 'dense_plus_bm25':
        return top_dense + (top_bm25 / max(top_bm25, 1.0))  # normalize bm25 contribution
    elif formula_name == 'fused_dense_raw':
        # Raw dense similarity of the top fused result
        return top_fused_dense
    elif formula_name == 'fused_dense_plus_margin':
        return top_fused_dense + margin * 2.0
    elif formula_name == 'dense_weighted_agreement':
        return top_dense + agreement * 0.1
    elif formula_name == 'dense_x_margin':
        return top_dense * (1.0 + margin)
    elif formula_name == 'evidence_score':
        # Combined evidence: high dense + some BM25 support + margin
        # Normalize BM25 into [0,1] range approximately (BM25 max is usually ~20-30 for good matches)
        bm25_norm = min(top_bm25 / 20.0, 1.0) if top_bm25 > 0 else 0.0
        return 0.6 * top_dense + 0.2 * bm25_norm + 0.2 * min(margin * 5.0, 1.0)
    elif formula_name == 'top5_overlap_dense':
        return top_dense + top5_overlap * 0.05
    elif formula_name == 'fused_score_raw':
        return fused
    elif formula_name == 'max_dense_margin':
        return max(top_dense, margin * 3.0)
    else:
        return fused


def test_combined_threshold(valid_data, no_match_data, formula_name, threshold):
    """Test a combined signal threshold."""
    nm_rejected = 0
    nm_total = len(no_match_data)
    for d in no_match_data:
        val = compute_combined_score(d, formula_name)
        if val < threshold:
            nm_rejected += 1
    
    v_rejected = 0
    v_total = len(valid_data)
    v_hits_1 = 0
    v_hits_5 = 0
    v_mrr_sum = 0.0
    
    for d in valid_data:
        val = compute_combined_score(d, formula_name)
        if val < threshold:
            v_rejected += 1
            continue
        if d['in_top_1']:
            v_hits_1 += 1
        if d['in_top_5']:
            v_hits_5 += 1
        if d['expected_fused_rank'] and d['expected_fused_rank'] > 0:
            v_mrr_sum += 1.0 / d['expected_fused_rank']
    
    nm_rejection_rate = nm_rejected / nm_total if nm_total > 0 else 0
    v_rejection_rate = v_rejected / v_total if v_total > 0 else 0
    fpr = (nm_total - nm_rejected) / nm_total if nm_total > 0 else 0
    fnr = v_rejected / v_total if v_total > 0 else 0
    hit_1 = v_hits_1 / v_total if v_total > 0 else 0
    hit_5 = v_hits_5 / v_total if v_total > 0 else 0
    mrr = v_mrr_sum / v_total if v_total > 0 else 0
    
    return {
        'formula': formula_name,
        'threshold': round(threshold, 6),
        'nm_rejected': nm_rejected,
        'nm_total': nm_total,
        'nm_rejection_rate': round(nm_rejection_rate, 4),
        'v_rejected': v_rejected,
        'v_total': v_total,
        'v_rejection_rate': round(v_rejection_rate, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'hit_at_1': round(hit_1, 4),
        'hit_at_5': round(hit_5, 4),
        'mrr': round(mrr, 4),
    }


# ========== MAIN ==========

def main():
    print("=" * 70)
    print("RECALL AI — PHASE 9 CONFIDENCE ANALYSIS")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Step 0: Verify frozen artifacts
    artifacts_ok, hashes = check_frozen_artifacts()
    if not artifacts_ok:
        print("CRITICAL: Frozen artifact integrity check FAILED. Aborting.")
        return
    
    # Load evaluation data
    test_queries = load_json('data/evaluation/test_queries_40.json')
    no_match_queries = load_json('data/evaluation/no_match_queries.json')
    print(f"Loaded {len(test_queries)} valid queries, {len(no_match_queries)} no-match queries.\n")
    
    # Initialize retrieval state
    print("Loading indexes...")
    t0 = time.time()
    br._ensure_bm25_state()
    br._ensure_dense_state()
    t1 = time.time()
    print(f"Indexes loaded in {(t1 - t0) * 1000:.0f} ms.\n")
    
    # ========== STEP 2: COLLECT CONFIDENCE DATA ==========
    print("=" * 70)
    print("STEP 2: COLLECTING CONFIDENCE DATA")
    print("=" * 70)
    
    valid_data = []
    no_match_data = []
    
    print("\nProcessing 40 valid queries...")
    for i, q_item in enumerate(test_queries):
        query = q_item['query']
        expected_id = q_item['answer_message_id']
        result = collect_confidence_data(query, expected_id, is_no_match=False)
        valid_data.append(result)
        hit_marker = "H1" if result['in_top_1'] else ("H5" if result['in_top_5'] else "--")
        print(f"  [{i+1:2d}/40] {hit_marker} dense={result['top_dense_raw']:.4f} "
              f"bm25={result['top_bm25_raw']:.2f} fused={result['top_fused_score']:.4f} "
              f"margin={result['fused_margin']:.4f} agree={result['bm25_dense_agreement']}")
    
    print("\nProcessing 10 no-match queries...")
    for i, q_item in enumerate(no_match_queries):
        query = q_item['query']
        result = collect_confidence_data(query, None, is_no_match=True)
        no_match_data.append(result)
        print(f"  [NM{i+1:2d}] dense={result['top_dense_raw']:.4f} "
              f"bm25={result['top_bm25_raw']:.2f} fused={result['top_fused_score']:.4f} "
              f"margin={result['fused_margin']:.4f} agree={result['bm25_dense_agreement']}")
    
    # Save raw results
    all_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'phase': '9A',
            'baseline': {'hit_at_1': 0.25, 'hit_at_5': 0.45, 'mrr': 0.3293, 'no_match_fpr': 1.0},
            'valid_count': len(valid_data),
            'no_match_count': len(no_match_data),
            'frozen_artifact_hashes': hashes,
        },
        'valid_queries': valid_data,
        'no_match_queries': no_match_data,
    }
    
    results_path = os.path.join(os.path.dirname(__file__), 'phase9_confidence_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRaw results saved to: {results_path}")
    
    # Verify baseline
    baseline_hits1 = sum(1 for d in valid_data if d['in_top_1'])
    baseline_hits5 = sum(1 for d in valid_data if d['in_top_5'])
    baseline_mrr = sum(1.0 / d['expected_fused_rank'] for d in valid_data if d['expected_fused_rank'] and d['expected_fused_rank'] > 0) / len(valid_data)
    print(f"\nBaseline verification:")
    print(f"  Hit@1 = {baseline_hits1}/40 = {baseline_hits1/40*100:.1f}%")
    print(f"  Hit@5 = {baseline_hits5}/40 = {baseline_hits5/40*100:.1f}%")
    print(f"  MRR   = {baseline_mrr:.4f}")
    
    # ========== STEP 3: SIGNAL ANALYSIS ==========
    print("\n" + "=" * 70)
    print("STEP 3: SIGNAL ANALYSIS — VALID vs NO-MATCH")
    print("=" * 70)
    
    signals_to_analyze = [
        ('top_fused_score', 'Top Fused Score'),
        ('top_dense_raw', 'Top Dense Score (absolute best)'),
        ('top_bm25_raw', 'Top BM25 Score (absolute best)'),
        ('fused_margin', 'Fused Score Margin (top1 - top2)'),
        ('top_fused_dense_raw', 'Dense Score of Top Fused Result'),
        ('top_fused_bm25_raw', 'BM25 Score of Top Fused Result'),
        ('bm25_dense_top5_overlap', 'BM25/Dense Top-5 Overlap Count'),
    ]
    
    signal_analysis_results = {}
    
    for signal_key, signal_label in signals_to_analyze:
        valid_vals = [d.get(signal_key, 0.0) or 0.0 for d in valid_data]
        nm_vals = [d.get(signal_key, 0.0) or 0.0 for d in no_match_data]
        
        analysis = analyze_signal_separation(valid_vals, nm_vals, signal_key)
        signal_analysis_results[signal_key] = analysis
        
        print(f"\n  {signal_label} ({signal_key}):")
        print(f"    VALID:    min={analysis['valid']['min']:.4f}  max={analysis['valid']['max']:.4f}  "
              f"mean={analysis['valid']['mean']:.4f}  median={analysis['valid']['median']:.4f}  "
              f"std={analysis['valid']['std']:.4f}  Q25={analysis['valid']['q25']:.4f}  Q75={analysis['valid']['q75']:.4f}")
        print(f"    NO-MATCH: min={analysis['no_match']['min']:.4f}  max={analysis['no_match']['max']:.4f}  "
              f"mean={analysis['no_match']['mean']:.4f}  median={analysis['no_match']['median']:.4f}  "
              f"std={analysis['no_match']['std']:.4f}  Q25={analysis['no_match']['q25']:.4f}  Q75={analysis['no_match']['q75']:.4f}")
        print(f"    Overlap:  {'YES — distributions overlap' if analysis['overlap_exists'] else 'NO — clean separation found!'}")
    
    # BM25/Dense agreement analysis (boolean)
    valid_agree = sum(1 for d in valid_data if d['bm25_dense_agreement'])
    nm_agree = sum(1 for d in no_match_data if d['bm25_dense_agreement'])
    print(f"\n  BM25/Dense Top-1 Agreement:")
    print(f"    VALID:    {valid_agree}/40 = {valid_agree/40*100:.1f}%")
    print(f"    NO-MATCH: {nm_agree}/10 = {nm_agree/10*100:.1f}%")
    
    # ========== STEP 4: THRESHOLD EXPERIMENTS ==========
    print("\n" + "=" * 70)
    print("STEP 4: THRESHOLD EXPERIMENTS")
    print("=" * 70)
    
    threshold_results = {}
    
    # For each promising signal, test a range of thresholds
    threshold_signals = [
        ('top_dense_raw', True),     # reject below
        ('top_fused_score', True),   # reject below
        ('fused_margin', True),      # reject below
        ('top_bm25_raw', True),      # reject below
        ('top_fused_dense_raw', True),
    ]
    
    for signal_key, reject_below in threshold_signals:
        valid_vals = sorted([d.get(signal_key, 0.0) or 0.0 for d in valid_data])
        nm_vals = sorted([d.get(signal_key, 0.0) or 0.0 for d in no_match_data])
        
        # Generate threshold candidates from the data
        all_vals = sorted(valid_vals + nm_vals)
        # Test at percentile boundaries and specific interesting points
        thresholds = sorted(set(
            [float(np.percentile(all_vals, p)) for p in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]] +
            [float(np.percentile(nm_vals, p)) for p in [25, 50, 75, 90, 95, 100]] +
            [float(np.percentile(valid_vals, p)) for p in [1, 5, 10, 25]]
        ))
        
        print(f"\n  Signal: {signal_key} (reject_below={reject_below})")
        print(f"  {'Threshold':>12}  {'NM_Rej':>6}  {'V_Rej':>5}  {'FPR':>6}  {'FNR':>6}  "
              f"{'Hit@1':>7}  {'Hit@5':>7}  {'MRR':>7}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
        
        signal_results = []
        for t in thresholds:
            r = test_threshold(valid_data, no_match_data, signal_key, t, reject_below)
            signal_results.append(r)
            # Only print interesting ones (where something changes)
            print(f"  {t:>12.6f}  {r['nm_rejected']:>4}/{r['nm_total']}  "
                  f"{r['v_rejected']:>3}/{r['v_total']}  {r['fpr']:>6.2%}  {r['fnr']:>6.2%}  "
                  f"{r['hit_at_1']:>6.1%}  {r['hit_at_5']:>6.1%}  {r['mrr']:>7.4f}")
        
        threshold_results[signal_key] = signal_results
    
    # ========== STEP 5: COMBINED SIGNAL EXPERIMENTS ==========
    print("\n" + "=" * 70)
    print("STEP 5: COMBINED SIGNAL EXPERIMENTS")
    print("=" * 70)
    
    combined_formulas = [
        'dense_plus_margin',
        'fused_dense_raw',
        'fused_dense_plus_margin',
        'dense_weighted_agreement',
        'dense_x_margin',
        'evidence_score',
        'top5_overlap_dense',
        'fused_score_raw',
        'max_dense_margin',
    ]
    
    combined_results = {}
    
    for formula in combined_formulas:
        valid_scores = [compute_combined_score(d, formula) for d in valid_data]
        nm_scores = [compute_combined_score(d, formula) for d in no_match_data]
        
        v_stats = compute_stats(valid_scores)
        nm_stats = compute_stats(nm_scores)
        
        print(f"\n  Formula: {formula}")
        print(f"    VALID:    min={v_stats['min']:.4f}  max={v_stats['max']:.4f}  mean={v_stats['mean']:.4f}  "
              f"median={v_stats['median']:.4f}  Q25={v_stats['q25']:.4f}  Q75={v_stats['q75']:.4f}")
        print(f"    NO-MATCH: min={nm_stats['min']:.4f}  max={nm_stats['max']:.4f}  mean={nm_stats['mean']:.4f}  "
              f"median={nm_stats['median']:.4f}  Q25={nm_stats['q25']:.4f}  Q75={nm_stats['q75']:.4f}")
        
        # Test thresholds for this combined signal
        all_scores = sorted(valid_scores + nm_scores)
        thresholds = sorted(set(
            [float(np.percentile(all_scores, p)) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]] +
            [float(np.percentile(nm_scores, p)) for p in [50, 75, 90, 95, 100]]
        ))
        
        print(f"    {'Threshold':>12}  {'NM_Rej':>6}  {'V_Rej':>5}  {'FPR':>6}  {'FNR':>6}  "
              f"{'Hit@1':>7}  {'Hit@5':>7}  {'MRR':>7}")
        
        formula_results = []
        for t in thresholds:
            r = test_combined_threshold(valid_data, no_match_data, formula, t)
            formula_results.append(r)
            print(f"    {t:>12.6f}  {r['nm_rejected']:>4}/{r['nm_total']}  "
                  f"{r['v_rejected']:>3}/{r['v_total']}  {r['fpr']:>6.2%}  {r['fnr']:>6.2%}  "
                  f"{r['hit_at_1']:>6.1%}  {r['hit_at_5']:>6.1%}  {r['mrr']:>7.4f}")
        
        combined_results[formula] = {
            'valid_stats': v_stats,
            'no_match_stats': nm_stats,
            'threshold_results': formula_results,
        }
    
    # ========== STEP 6: RETRIEVAL IMPACT SUMMARY ==========
    print("\n" + "=" * 70)
    print("STEP 6: RETRIEVAL IMPACT — BEST CANDIDATES")
    print("=" * 70)
    
    print("\nBaseline (Phase 8, no rejection):")
    print(f"  Hit@1 = {baseline_hits1/40*100:.1f}%  Hit@5 = {baseline_hits5/40*100:.1f}%  MRR = {baseline_mrr:.4f}  FPR = 100%")
    
    # Find promising strategies: those with >=50% no-match rejection and <=10% valid rejection
    print("\n  Promising strategies (NM rejection >= 30%, valid rejection <= 15%):")
    
    best_candidates = []
    
    # Check individual signals
    for signal_key, results in threshold_results.items():
        for r in results:
            if r['nm_rejection_rate'] >= 0.30 and r['v_rejection_rate'] <= 0.15:
                best_candidates.append(r)
                print(f"    {r['signal']} @ {r['threshold']:.6f}: "
                      f"NM_rej={r['nm_rejected']}/{r['nm_total']}  "
                      f"V_rej={r['v_rejected']}/{r['v_total']}  "
                      f"FPR={r['fpr']:.0%}  FNR={r['fnr']:.0%}  "
                      f"Hit@1={r['hit_at_1']:.1%}  Hit@5={r['hit_at_5']:.1%}  MRR={r['mrr']:.4f}")
    
    # Check combined signals
    for formula, data in combined_results.items():
        for r in data['threshold_results']:
            if r['nm_rejection_rate'] >= 0.30 and r['v_rejection_rate'] <= 0.15:
                best_candidates.append(r)
                print(f"    {r['formula']} @ {r['threshold']:.6f}: "
                      f"NM_rej={r['nm_rejected']}/{r['nm_total']}  "
                      f"V_rej={r['v_rejected']}/{r['v_total']}  "
                      f"FPR={r['fpr']:.0%}  FNR={r['fnr']:.0%}  "
                      f"Hit@1={r['hit_at_1']:.1%}  Hit@5={r['hit_at_5']:.1%}  MRR={r['mrr']:.4f}")
    
    if not best_candidates:
        print("    NONE found. No strategy achieves >=30% NM rejection with <=15% valid rejection.")
    
    # ========== STEP 7: OVERFITTING WARNING ==========
    print("\n" + "=" * 70)
    print("STEP 7: OVERFITTING ASSESSMENT")
    print("=" * 70)
    print("  Dataset: 40 valid + 10 no-match = 50 total queries")
    print("  This dataset is TOO SMALL for production-grade calibration.")
    print("  Any threshold found here is an evaluation-set observation only.")
    print("  A legitimate train/validation split is NOT possible with 10 no-match queries.")
    print("  Leave-one-out cross-validation would still only have 9 no-match training samples.")
    print("  Therefore: NO classifier should be trained on this dataset.")
    print("  All threshold findings are DESCRIPTIVE, not PREDICTIVE.")
    
    # ========== STEP 8: CROSS-ENCODER ASSESSMENT ==========
    print("\n" + "=" * 70)
    print("STEP 8: CROSS-ENCODER ASSESSMENT")
    print("=" * 70)
    
    has_promising = len(best_candidates) > 0
    if has_promising:
        print("  Current signals show SOME separation potential.")
        print("  Cross-encoder may not be necessary for Phase 9.")
        print("  However, cross-encoder could provide better no-match signal:")
        print("    - A cross-encoder scores (query, document) pairs directly")
        print("    - Suitable model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("    - Alternative: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("    - Expected runtime: ~50-100ms per candidate (top-5 = 250-500ms)")
        print("    - Would add ~300MB model download")
        print("    - Practical for reranking top-N candidates only")
    else:
        print("  Current signals do NOT reliably separate valid from no-match queries.")
        print("  A cross-encoder COULD potentially help:")
        print("    - Cross-encoders compute direct (query, document) relevance")
        print("    - Suitable multilingual model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("    - Alternative: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("    - Expected cost: ~50-100ms per candidate pair")
        print("    - For top-5 reranking: ~250-500ms additional latency")
        print("    - Would add ~300MB model download")
        print("    - Could provide a direct relevance score that better separates classes")
        print("  RECOMMENDATION: Investigate in Phase 9B if approved.")
    
    # ========== PERFORMANCE MEASUREMENT ==========
    print("\n" + "=" * 70)
    print("PERFORMANCE MEASUREMENT")
    print("=" * 70)
    
    timings = {}
    queries_to_time = {
        'semantic': "What did we decide about the trip?",
        'attributed': "What did Priya say about budget?",
        'temporal': "What did we discuss in June 2026?",
        'mixed': "What did Priya say about budget in June 2026?",
        'no_match': "Did anyone figure out the processing time for the Schengen visa?",
    }
    
    for q_type, q_text in queries_to_time.items():
        times = []
        for _ in range(5):
            t_start = time.time()
            hybrid_search(q_text, top_k=5)
            t_end = time.time()
            times.append((t_end - t_start) * 1000)
        avg_time = mean(times)
        timings[q_type] = round(avg_time, 2)
        print(f"  {q_type:>12}: {avg_time:.2f} ms (avg of 5 runs)")
    
    # ========== SAVE COMPLETE ANALYSIS ==========
    complete_analysis = {
        'metadata': all_data['metadata'],
        'frozen_artifact_hashes': hashes,
        'baseline_verification': {
            'hit_at_1': f"{baseline_hits1}/40 = {baseline_hits1/40*100:.1f}%",
            'hit_at_5': f"{baseline_hits5}/40 = {baseline_hits5/40*100:.1f}%",
            'mrr': round(baseline_mrr, 4),
        },
        'signal_analysis': {k: {
            'valid': v['valid'],
            'no_match': v['no_match'],
            'overlap_exists': v['overlap_exists'],
        } for k, v in signal_analysis_results.items()},
        'threshold_experiments': {k: v for k, v in threshold_results.items()},
        'combined_signal_experiments': combined_results,
        'promising_strategies': best_candidates,
        'performance_ms': timings,
    }
    
    analysis_path = os.path.join(os.path.dirname(__file__), 'phase9_confidence_results.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(complete_analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nComplete analysis saved to: {analysis_path}")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("PHASE 9A - FINAL SUMMARY")
    print("=" * 70)
    print(f"  Baseline: Hit@1={baseline_hits1/40*100:.1f}%  Hit@5={baseline_hits5/40*100:.1f}%  MRR={baseline_mrr:.4f}  FPR=100%")
    print(f"  Signals analyzed: {len(signals_to_analyze)} individual + {len(combined_formulas)} combined")
    print(f"  Promising strategies found: {len(best_candidates)}")
    
    if best_candidates:
        # Find the best candidate (highest NM rejection with acceptable valid loss)
        best = max(best_candidates, key=lambda x: (x['nm_rejection_rate'], -x['v_rejection_rate'], x['mrr']))
        strategy_key = best.get('signal', best.get('formula', 'unknown'))
        print(f"\n  BEST CANDIDATE:")
        print(f"    Strategy: {strategy_key} @ threshold={best['threshold']:.6f}")
        print(f"    NM rejection: {best['nm_rejected']}/{best['nm_total']} ({best['nm_rejection_rate']:.0%})")
        print(f"    Valid rejection: {best['v_rejected']}/{best['v_total']} ({best['v_rejection_rate']:.0%})")
        print(f"    FPR: {best['fpr']:.0%}  FNR: {best['fnr']:.0%}")
        print(f"    Hit@1: {best['hit_at_1']:.1%}  Hit@5: {best['hit_at_5']:.1%}  MRR: {best['mrr']:.4f}")
        print(f"\n  RECOMMENDATION: A confidence strategy shows promise. Proceed to Phase 9B implementation.")
    else:
        print(f"\n  RECOMMENDATION: No reliable confidence strategy found.")
        print(f"  Preserve Phase 8 baseline. Document no-match as unresolved.")
    
    print(f"\nPhase 9A analysis completed at: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
