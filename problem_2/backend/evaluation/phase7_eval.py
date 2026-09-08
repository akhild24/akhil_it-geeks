import json
import time
import os
import sys

# Add parent directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import hybrid_search
import backend.retrieval.basic_retrieval as br

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

test_queries = load_json('data/evaluation/test_queries_40.json')
no_match_queries = load_json('data/evaluation/no_match_queries.json')

print("Loading indexes...")
t_load_start = time.time()
br._ensure_bm25_state()
t_load_end = time.time()
print(f"Indexes loaded in {(t_load_end - t_load_start)*1000:.2f} ms")

# Evaluate 40 queries
success_count = 0
fail_count = 0
ranks = []
hits_at_1 = 0
hits_at_5 = 0

type_metrics = {
    'semantic': {'count': 0, 'hits_at_1': 0, 'hits_at_5': 0},
    'attributed': {'count': 0, 'hits_at_1': 0, 'hits_at_5': 0},
    'temporal': {'count': 0, 'hits_at_1': 0, 'hits_at_5': 0},
    'mixed': {'count': 0, 'hits_at_1': 0, 'hits_at_5': 0},
    'ambiguous': {'count': 0, 'hits_at_1': 0, 'hits_at_5': 0}
}

print("\nEvaluating 40 gold queries...")
for q_item in test_queries:
    query = q_item['query']
    expected_id = q_item['answer_message_id']
    
    cls = classify_query(query)
    q_type = cls.query_type
    
    results = hybrid_search(query, top_k=100)
    
    rank = -1
    for i, r in enumerate(results):
        if r['message_id'] == expected_id:
            rank = i + 1
            break
            
    if rank != -1:
        ranks.append(rank)
        success_count += 1
        if rank == 1:
            hits_at_1 += 1
            type_metrics[q_type]['hits_at_1'] += 1
        if rank <= 5:
            hits_at_5 += 1
            type_metrics[q_type]['hits_at_5'] += 1
    else:
        ranks.append(0)
        fail_count += 1
        
    type_metrics[q_type]['count'] += 1

total = len(test_queries)
p_at_1 = hits_at_1 / total
# P@5 for 1 answer per query is (1/5) if Hit@5 is true
p_at_5 = (hits_at_5 / total) * (1/5)
r_at_1 = hits_at_1 / total
r_at_5 = hits_at_5 / total
mrr = sum([1/r for r in ranks if r > 0]) / total

print("\n--- OVERALL METRICS ---")
print(f"Total Queries: {total}")
print(f"Successful: {success_count}, Failed: {fail_count}")
print(f"Hit@1: {hits_at_1} ({(r_at_1*100):.1f}%)")
print(f"Hit@5: {hits_at_5} ({(r_at_5*100):.1f}%)")
print(f"Precision@1: {p_at_1:.4f}")
print(f"Precision@5: {p_at_5:.4f}")
print(f"Recall@1: {r_at_1:.4f}")
print(f"Recall@5: {r_at_5:.4f}")
print(f"MRR: {mrr:.4f}")

print("\n--- BY TYPE ---")
for t, m in type_metrics.items():
    if m['count'] > 0:
        print(f"{t.upper()}: Count={m['count']}, Hit@1={m['hits_at_1']}, Hit@5={m['hits_at_5']}")

print("\n--- NO MATCH EVALUATION ---")
fpr = 0
for q_item in no_match_queries:
    q = q_item['query']
    results = hybrid_search(q, top_k=1)
    if results:
        fpr += 1

print(f"No match queries: {len(no_match_queries)}")
print(f"False Positive Rate (returns results when shouldn't): {fpr}/{len(no_match_queries)} ({(fpr/len(no_match_queries)*100):.1f}%)")
print("NOTE: Threshold intentionally omitted in Phase 5 due to RRF score overlap.")

print("\n--- PERFORMANCE ---")
t0 = time.time()
hybrid_search("What did we decide about the trip?", top_k=5)
t1 = time.time()
hybrid_search("What did Priya say about budget?", top_k=5)
t2 = time.time()
hybrid_search("What did we discuss in June 2026?", top_k=5)
t3 = time.time()
hybrid_search("What did Priya say about budget in June 2026?", top_k=5)
t4 = time.time()

print(f"Semantic search: {(t1-t0)*1000:.2f} ms")
print(f"Attributed search: {(t2-t1)*1000:.2f} ms")
print(f"Temporal search: {(t3-t2)*1000:.2f} ms")
print(f"Mixed search: {(t4-t3)*1000:.2f} ms")
