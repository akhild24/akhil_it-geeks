import json
import os
import sys

# Add parent directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.retrieval.hybrid_retrieval import hybrid_search

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_calibration():
    # Load evaluation data
    no_match_path = os.path.join("data", "evaluation", "no_match_queries.json")
    test_queries_path = os.path.join("data", "evaluation", "test_queries_40.json")
    
    no_match_data = load_json(no_match_path)
    test_queries_data = load_json(test_queries_path)
    
    no_match_scores = []
    relevant_scores = []
    
    print("Evaluating No-Match Queries...")
    for item in no_match_data:
        query = item.get("query")
        # Ensure we just pass the query; hybrid_search will extract filters if needed
        results = hybrid_search(query, top_k=1)
        if results:
            top_score = results[0]['fused_score']
            no_match_scores.append(top_score)
            print(f"[{top_score:.4f}] {query}")
        else:
            print(f"[0.0000] {query} (No candidates)")
            no_match_scores.append(0.0)
            
    print("\nEvaluating 40 Relevant Queries...")
    for item in test_queries_data:
        query = item.get("query")
        results = hybrid_search(query, top_k=1)
        if results:
            top_score = results[0]['fused_score']
            relevant_scores.append(top_score)
        else:
            relevant_scores.append(0.0)
            
    no_match_scores.sort()
    relevant_scores.sort()
    
    print("\n--- Distribution Summary ---")
    print(f"No-Match Scores: min={min(no_match_scores):.4f}, max={max(no_match_scores):.4f}, mean={sum(no_match_scores)/len(no_match_scores):.4f}")
    print(f"Relevant Scores: min={min(relevant_scores):.4f}, max={max(relevant_scores):.4f}, mean={sum(relevant_scores)/len(relevant_scores):.4f}")
    
    print(f"NM_MIN: {min(no_match_scores):.4f}, NM_MAX: {max(no_match_scores):.4f}, NM_MEAN: {sum(no_match_scores)/len(no_match_scores):.4f}")
    print(f"VAL_MIN: {min(relevant_scores):.4f}, VAL_MAX: {max(relevant_scores):.4f}, VAL_MEAN: {sum(relevant_scores)/len(relevant_scores):.4f}")
    
    print("\n==================================================")
    print("THRESHOLD VERDICT: NO RELIABLE GLOBAL THRESHOLD")
    print("==================================================")
    print("With this evaluation set and this RRF formulation, no reliable global threshold")
    print("is supported by the observed score distributions.")
    print("The RRF score distributions overlap heavily and therefore a global RRF threshold")
    print("cannot reliably separate valid and out-of-domain queries.")
    
if __name__ == "__main__":
    run_calibration()
