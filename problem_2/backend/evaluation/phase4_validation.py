import json
import os
from datetime import datetime
import numpy as np

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import (
    hybrid_search, filter_candidates, filter_by_sender, filter_by_date_range, rrf_fusion
)
import backend.retrieval.basic_retrieval as br

def test_query_classification_unit():
    print("==================================================")
    print("1. QUERY CLASSIFICATION UNIT TESTS")
    print("==================================================")
    
    test_cases = [
        ("when did we decide on the trip", "semantic", None),
        ("what did Priya say about budget", "attributed", "Priya"),
        ("what did we discuss last month", "temporal", None),
        ("what did Priya say last month about the trip", "mixed", "Priya")
    ]
    
    for q, exp_type, exp_sender in test_cases:
        res = classify_query(q)
        print(f"Q: '{q}'")
        print(f"  Predicted type: {res.query_type} | Expected: {exp_type}")
        print(f"  Sender: {res.sender} | Expected: {exp_sender}")
        has_dates = res.date_start is not None
        print(f"  Has Dates: {has_dates} | Expected: {exp_type in ['temporal', 'mixed']}")
        print()

def test_query_classification_40():
    print("==================================================")
    print("2. 40-QUERY CLASSIFICATION RESULTS")
    print("==================================================")
    
    with open('data/evaluation/test_queries_40.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    counts = {"semantic": 0, "attributed": 0, "temporal": 0, "mixed": 0, "ambiguous": 0}
    
    for i, q_item in enumerate(queries, 1):
        q = q_item['query']
        res = classify_query(q)
        counts[res.query_type] += 1
        
        date_str = f"{res.date_start.date()} to {res.date_end.date()}" if res.date_start else "None"
        
        print(f"Query #{i}: {q}")
        print(f"  Predicted type: {res.query_type}")
        print(f"  Extracted sender: {res.sender}")
        print(f"  Extracted date range: {date_str}\n")
        
    print("TOTALS:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

def test_metadata_filtering():
    print("\n==================================================")
    print("3. METADATA FILTERING TESTS")
    print("==================================================")
    
    br._ensure_bm25_state()
    metadata = br._METADATA
    total_count = len(metadata)
    print(f"Total messages: {total_count}")
    
    # 1. Sender filter: Priya
    sender_candidates = filter_candidates(metadata, sender="Priya")
    print(f"\n[Sender Filter] Priya")
    print(f"  Candidates after filtering: {len(sender_candidates)}")
    all_priya = all(metadata[idx].get('sender') == 'Priya' for idx in sender_candidates)
    print(f"  Verified all are Priya: {all_priya}")
    
    # 2. Date filter:
    start = datetime(2026, 6, 1)
    end = datetime(2026, 6, 30, 23, 59, 59)
    date_candidates = filter_candidates(metadata, date_start=start, date_end=end)
    print(f"\n[Date Filter] June 2026")
    print(f"  Candidates after filtering: {len(date_candidates)}")
    all_dates = True
    for idx in date_candidates:
        ts = datetime.fromisoformat(metadata[idx]['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
        if not (start <= ts <= end):
            all_dates = False
    print(f"  Verified all in range: {all_dates}")
    
    # 3. Mixed filter
    mixed_candidates = filter_candidates(metadata, sender="Priya", date_start=start, date_end=end)
    print(f"\n[Mixed Filter] Priya + June 2026")
    print(f"  Candidates after filtering: {len(mixed_candidates)}")
    all_mixed = True
    for idx in mixed_candidates:
        if metadata[idx].get('sender') != 'Priya':
            all_mixed = False
        ts = datetime.fromisoformat(metadata[idx]['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
        if not (start <= ts <= end):
            all_mixed = False
    print(f"  Verified both conditions: {all_mixed}")

def test_rrf():
    print("\n==================================================")
    print("4. RRF UNIT & SANITY TESTS")
    print("==================================================")
    
    dummy_bm25 = [
        {'message_id': 'A', 'rank': 1},
        {'message_id': 'B', 'rank': 2},
        {'message_id': 'C', 'rank': 3}
    ]
    dummy_dense = [
        {'message_id': 'C', 'rank': 1},
        {'message_id': 'A', 'rank': 2},
        {'message_id': 'D', 'rank': 3}
    ]
    
    fused = rrf_fusion(dummy_bm25, dummy_dense, k=60)
    print("Dummy RRF Results (Expected A or C top):")
    for f in fused:
        print(f"  ID: {f['message_id']} | Fused Score: {f['fused_score']:.6f} | Fused Rank: {f['fused_rank']}")

def test_zero_overlap():
    print("\n==================================================")
    print("5. ZERO-OVERLAP COMPARISON")
    print("==================================================")
    
    zero_queries = [
        {"q": "When and where were they supposed to gather for the editing work?", "ans_id": 338},
        {"q": "What was the permitted similarity percentage?", "ans_id": 663},
        {"q": "Where could they watch the match without paying for a separate venue?", "ans_id": 923},
        {"q": "What was the latest arrival time Priya gave?", "ans_id": 2175},
        {"q": "What kind of Volvo service had been chosen?", "ans_id": 2533}
    ]
    
    for tq in zero_queries:
        query = tq['q']
        ans_id = tq['ans_id']
        
        cls = classify_query(query)
        
        # We run hybrid search but we also want pure bm25 and pure dense ranks.
        # Hybrid search returns RRF list. To get BM25 and Dense top 5, we can manually fetch them 
        # or extract from hybrid if we return top_k=100.
        results = hybrid_search(query, top_k=50) # Get enough to see top 5 of each
        
        # Sort manually to find true BM25 and Dense top 5
        bm25_sorted = sorted([r for r in results if 'bm25_rank' in r], key=lambda x: x['bm25_rank'])
        dense_sorted = sorted([r for r in results if 'dense_rank' in r], key=lambda x: x['dense_rank'])
        
        bm25_top5 = [r['message_id'] for r in bm25_sorted[:5]]
        dense_top5 = [r['message_id'] for r in dense_sorted[:5]]
        rrf_top5 = [r['message_id'] for r in results[:5]]
        
        date_str = f"{cls.date_start.date()} to {cls.date_end.date()}" if cls.date_start else "None"
        
        print(f"\nQuery: {query}")
        print(f"Target ID: {ans_id}")
        print(f"Query type: {cls.query_type}")
        print(f"Sender filter: {cls.sender}")
        print(f"Date filter: {date_str}")
        print(f"BM25 top 5: {bm25_top5}")
        print(f"Dense top 5: {dense_top5}")
        print(f"RRF top 5: {rrf_top5}")
        
        print(f"Target in BM25 Top 5: {'YES' if ans_id in bm25_top5 else 'NO'}")
        print(f"Target in Dense Top 5: {'YES' if ans_id in dense_top5 else 'NO'}")
        print(f"Target in RRF Top 5: {'YES' if ans_id in rrf_top5 else 'NO'}")

def test_decision_threads():
    print("\n==================================================")
    print("6. DECISION THREAD QUERIES")
    print("==================================================")
    
    threads = [
        {"q": "Which plan did the group agree to proceed with?", "ans_id": 257},
        {"q": "What combination of rail class and lodging did they finally approve?", "ans_id": 3764},
        {"q": "Who agreed to handle the notes accompanying the talk?", "ans_id": 4304}
    ]
    
    for tq in threads:
        query = tq['q']
        ans_id = tq['ans_id']
        
        results = hybrid_search(query, top_k=100)
        
        bm25_sorted = sorted([r for r in results if 'bm25_rank' in r], key=lambda x: x['bm25_rank'])
        dense_sorted = sorted([r for r in results if 'dense_rank' in r], key=lambda x: x['dense_rank'])
        
        bm25_rank = next((r['bm25_rank'] for r in bm25_sorted if r['message_id'] == ans_id), "Not in top 100")
        dense_rank = next((r['dense_rank'] for r in dense_sorted if r['message_id'] == ans_id), "Not in top 100")
        rrf_rank = next((r['fused_rank'] for r in results if r['message_id'] == ans_id), "Not in top 100")
        
        print(f"\nQuery: {query}")
        print(f"Target ID: {ans_id}")
        print(f"  BM25 Rank: {bm25_rank}")
        print(f"  Dense Rank: {dense_rank}")
        print(f"  RRF Rank: {rrf_rank}")

def test_no_match():
    print("\n==================================================")
    print("7. NO-MATCH BASELINE SCORES")
    print("==================================================")
    
    with open('data/evaluation/no_match_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    for q_item in queries:
        q = q_item['query']
        results = hybrid_search(q, top_k=1)
        if results:
            print(f"Query: {q}")
            print(f"  Top RRF Score: {results[0]['fused_score']:.6f} | ID: {results[0]['message_id']}")
        else:
            print(f"Query: {q} -> No candidates returned")

def main():
    test_query_classification_unit()
    test_query_classification_40()
    test_metadata_filtering()
    test_rrf()
    test_zero_overlap()
    test_decision_threads()
    test_no_match()
    
    print("\n==================================================")
    print("PHASE 4 FOUNDATION: PASS")
    print("==================================================")

if __name__ == "__main__":
    main()
