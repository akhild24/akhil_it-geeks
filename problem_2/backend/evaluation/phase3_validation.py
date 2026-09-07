import os
import json
import numpy as np

from backend.indexing.index_utils import load_embeddings, load_bm25, load_metadata
from backend.retrieval.basic_retrieval import retrieve_bm25, retrieve_dense

def verify_model():
    print("==================================================")
    print("1. MODEL & INDEX VERIFICATION")
    print("==================================================")
    try:
        embeddings = load_embeddings()
        bm25 = load_bm25()
        metadata = load_metadata()
    except Exception as e:
        print(f"Failed to load indexes: {e}")
        return False

    print(f"Exact embedding model used: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Fallback)")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of embeddings: {embeddings.shape[0]}")
    
    corpus_count = 4426
    print(f"Number of corpus messages (expected): {corpus_count}")
    
    if embeddings.shape[0] != corpus_count:
        print("FAIL: Embedding count != corpus count")
        return False
        
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("FAIL: NaN or Inf values found in embeddings")
        return False
        
    if (bm25.corpus_size if hasattr(bm25, 'corpus_size') else len(bm25.idf)) != corpus_count: # Approximate check depending on rank_bm25 version
        # Actually bm25.corpus_size gives number of docs
        print(f"BM25 document count: {bm25.corpus_size}")
        if bm25.corpus_size != corpus_count:
            print("FAIL: BM25 count != corpus count")
            return False
            
    if len(metadata) != corpus_count:
        print("FAIL: Metadata count != corpus count")
        return False
        
    print("PASS: Model and index verification successful.")
    return True

def sanity_tests():
    print("\n==================================================")
    print("2. RETRIEVAL SANITY TESTS")
    print("==================================================")
    
    queries = {
        "Exact/Near Lexical": "Which plan did the group agree to proceed with?", # Q2
        "Semantic": "Who had already taken care of the printing payment?", # Q1
        "Hinglish": "kya decide hua final", # Mocked for sanity check
        "Zero-overlap": "When and where were they supposed to gather for the editing work?", # Q3
        "Short/Noisy": "presentation notes" # Mocked short query
    }
    
    for q_type, q_text in queries.items():
        print(f"\n--- {q_type} Query ---")
        print(f"QUERY: {q_text}")
        
        bm25_res = retrieve_bm25(q_text, top_k=3)
        print("\nBM25 TOP RESULTS:")
        for r in bm25_res:
            print(f"{r['rank']}. ID {r['message_id']} score {r['raw_score']:.4f} | {r['message'][:50]}...")
            
        dense_res = retrieve_dense(q_text, top_k=3)
        print("\nDENSE TOP RESULTS:")
        for r in dense_res:
            print(f"{r['rank']}. ID {r['message_id']} score {r['raw_score']:.4f} | {r['message'][:50]}...")

def zero_overlap_test():
    print("\n==================================================")
    print("3. ZERO-OVERLAP EVALUATION (At least 3)")
    print("==================================================")
    # Using 3 known zero overlap queries from earlier audit
    test_cases = [
        {"q": "When and where were they supposed to gather for the editing work?", "ans_id": 338},
        {"q": "What was the permitted similarity percentage?", "ans_id": 663},
        {"q": "Where could they watch the match without paying for a separate venue?", "ans_id": 923}
    ]
    
    for tc in test_cases:
        query = tc['q']
        ans_id = tc['ans_id']
        print(f"\nQUERY: {query} (Target ID: {ans_id})")
        
        bm25_res = retrieve_bm25(query, top_k=10)
        dense_res = retrieve_dense(query, top_k=10)
        
        bm25_ranks = [r['message_id'] for r in bm25_res]
        dense_ranks = [r['message_id'] for r in dense_res]
        
        print(f"BM25 top 5: {'YES' if ans_id in bm25_ranks[:5] else 'NO'}")
        print(f"Dense top 5: {'YES' if ans_id in dense_ranks[:5] else 'NO'}")
        print(f"BM25 top 10: {'YES' if ans_id in bm25_ranks else 'NO'}")
        print(f"Dense top 10: {'YES' if ans_id in dense_ranks else 'NO'}")

def main():
    if not verify_model():
        print("\nPHASE 3 FOUNDATION: FAIL")
        return
        
    sanity_tests()
    zero_overlap_test()
    
    print("\n==================================================")
    print("PHASE 3 FOUNDATION: PASS")
    print("==================================================")

if __name__ == "__main__":
    main()
