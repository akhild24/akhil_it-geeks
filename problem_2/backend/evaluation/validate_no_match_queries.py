import json
import os
import re

CORPUS_PATH = r"c:\Users\ASUS\OneDrive\Desktop\akhil\data\chat_corpus_2.json"
QUERIES_PATH = r"c:\Users\ASUS\OneDrive\Desktop\akhil\data\evaluation\no_match_queries.json"

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

def validate():
    print("==================================================")
    print("NO-MATCH EVALUATION DATASET VALIDATION")
    print("==================================================")
    
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    print(f"Loaded {len(corpus)} messages from Corpus 2.")
    print(f"Loaded {len(queries)} queries from no_match_queries.json.\n")
    
    num_queries = len(queries)
    if 8 <= num_queries <= 10:
        print(f"[PASS] Number of queries: {num_queries} (Required: 8-10)")
    else:
        print(f"[FAIL] Number of queries: {num_queries} (Required: 8-10)")
        
    seen_texts = set()
    duplicates = 0
    invalid_format = 0
    empty_queries = 0
    failed_queries = 0
    
    # Common stop words to exclude from lexical check
    stop_words = {"the", "is", "in", "to", "for", "of", "a", "an", "and", "or", "did", "we", "they", "who", "what", "how", "are", "it", "this", "that", "on", "at", "anyone", "everyone", "someone", "from", "with", "have", "has", "had", "was", "were", "do", "does", "done", "out"}
    
    print("--------------------------------------------------")
    for i, q in enumerate(queries):
        query_text = q.get('query', '')
        print(f"Query {i+1}: {query_text}")
        
        # Check empty
        if not query_text.strip():
            print("  [FAIL] Query is empty.")
            empty_queries += 1
            
        # Check duplicates
        if query_text in seen_texts:
            print("  [FAIL] Duplicate query.")
            duplicates += 1
        seen_texts.add(query_text)
        
        # Check format
        if q.get('answer_message_id') is not None or q.get('expected_no_match') is not True:
            print("  [FAIL] Invalid format (answer_message_id must be null, expected_no_match must be true).")
            invalid_format += 1
            
        # Lexical evidence check
        q_tokens = normalize_text(query_text) - stop_words
        evidence_found = False
        relevant_ids = []
        
        # The goal is to ensure NO message contains the *core topics* of the query.
        # So we look for strong overlap of rare/important words.
        for msg in corpus:
            msg_tokens = normalize_text(msg.get('text', ''))
            overlap = q_tokens.intersection(msg_tokens)
            # If a single message contains more than 1 or 2 significant topical words, we flag it.
            # To be very strict, we flag if the core noun/entity overlaps significantly.
            # We already chose topics that have 0 hits, so the core topic words won't appear.
            # But just in case:
            if len(overlap) >= 3:
                evidence_found = True
                relevant_ids.append(msg['id'])
        
        if evidence_found:
            print(f"  [FAIL] Lexical evidence found: YES (Message IDs: {relevant_ids[:5]}...)")
            print("  Final no-match validation: FAIL")
            failed_queries += 1
        else:
            print("  [PASS] Lexical evidence found: NO")
            print("  Final no-match validation: PASS")
        print("--------------------------------------------------")
        
    print("\n[NOTE] Semantic validation will be confirmed again during Phase 3 retrieval evaluation.")
    print("==================================================")
    
    overall_pass = (
        8 <= num_queries <= 10 and 
        empty_queries == 0 and 
        duplicates == 0 and 
        invalid_format == 0 and 
        failed_queries == 0
    )
    print(f"OVERALL NO-MATCH VALIDATION: {'PASS' if overall_pass else 'FAIL'}")

if __name__ == "__main__":
    validate()
