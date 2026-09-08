import json
import os
import re
from datetime import datetime

def validate_corpus(filepath):
    print(f"Loading corpus from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    num_messages = len(corpus)
    
    senders = set()
    invalid_ids = 0
    broken_links = 0
    invalid_timestamps = 0
    empty_messages = 0
    forwarded_messages = 0
    short_replies = 0
    
    seen_ids = set()
    
    # Check IDs and collect data
    for msg in corpus:
        if msg.get("id") is None:
            invalid_ids += 1
        elif msg["id"] in seen_ids:
            invalid_ids += 1
        else:
            seen_ids.add(msg["id"])
            
        if not msg.get("text") or str(msg["text"]).strip() == "":
            empty_messages += 1
        if "[Forwarded]" in str(msg.get("text", "")):
            forwarded_messages += 1
        if len(str(msg.get("text", "")).strip()) <= 5:
            short_replies += 1
            
        senders.add(msg.get("sender"))
        
    # Check links and timestamps
    start_date = None
    end_date = None
    
    for i, msg in enumerate(corpus):
        try:
            ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            if start_date is None or ts < start_date:
                start_date = ts
            if end_date is None or ts > end_date:
                end_date = ts
        except Exception:
            invalid_timestamps += 1
            
        # Check chronological order locally
        if i > 0:
            prev_ts = datetime.fromisoformat(corpus[i-1]["timestamp"].replace("Z", "+00:00"))
            ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            if ts < prev_ts:
                invalid_timestamps += 1
                
        # Link check
        if msg.get("prev_id") is not None:
            if msg["prev_id"] not in seen_ids:
                broken_links += 1
        if msg.get("next_id") is not None:
            if msg["next_id"] not in seen_ids:
                broken_links += 1
                
        if i > 0 and msg.get("prev_id") != corpus[i-1]["id"]:
            broken_links += 1
        if i < len(corpus) - 1 and msg.get("next_id") != corpus[i+1]["id"]:
            broken_links += 1

    # Date validation
    current_date = datetime.utcnow().replace(tzinfo=None)
    future_dates = 0
    if end_date and end_date.replace(tzinfo=None) > current_date:
        future_dates = 1
        
    duration = (end_date - start_date).days if (end_date and start_date) else 0

    status = "PASS"
    if num_messages < 4000: status = "FAIL (Not enough messages)"
    if len(senders) != 8: status = "FAIL (Participants mismatch)"
    if invalid_ids > 0: status = "FAIL (Invalid IDs)"
    if broken_links > 0: status = "FAIL (Broken links)"
    if invalid_timestamps > 0: status = "FAIL (Invalid timestamps)"
    if empty_messages > 0: status = "FAIL (Empty messages)"
    if future_dates > 0: status = "FAIL (Future timestamps detected)"
    if not (150 <= duration <= 210): status = f"FAIL (Duration {duration} days is not approx 6 months)"
    
    print("\n## CORPUS 2 VALIDATION")
    print("-" * 25)
    print(f"Messages: {num_messages}")
    print(f"Participants ({len(senders)}): {list(senders)}")
    print(f"Start date: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'}")
    print(f"End date: {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}")
    print(f"Duration: approximately {duration // 30} months ({duration} days)")
    print(f"Future dates: {'Yes' if future_dates > 0 else 'No'}")
    print(f"Invalid IDs: {invalid_ids}")
    print(f"Broken links: {broken_links}")
    print(f"Invalid timestamps: {invalid_timestamps}")
    print(f"Empty messages: {empty_messages}")
    print(f"Forwarded messages: {forwarded_messages}")
    print(f"One-word/short replies: {short_replies}")
    print("-" * 25)
    print(f"Status: {status}")
    return corpus, seen_ids

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

def validate_queries(queries_filepath, corpus, valid_ids):
    print(f"\nLoading queries from {queries_filepath}...")
    with open(queries_filepath, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    num_queries = len(queries)
    
    empty_queries = 0
    invalid_answer_ids = 0
    duplicates = 0
    
    seen_queries = set()
    zero_overlap_count = 0
    query_types = set()
    
    # lookup corpus text by id
    corpus_dict = {msg["id"]: msg["text"] for msg in corpus}
    
    for q in queries:
        qt = q.get("query", "")
        if not qt or str(qt).strip() == "":
            empty_queries += 1
            
        if qt in seen_queries:
            duplicates += 1
        else:
            seen_queries.add(qt)
            
        ans_id = q.get("answer_message_id")
        if ans_id not in valid_ids:
            invalid_answer_ids += 1
            
        if ans_id in corpus_dict and qt.strip():
            ans_text = corpus_dict[ans_id]
            q_tokens = tokenize(qt)
            a_tokens = tokenize(ans_text)
            
            overlap = q_tokens.intersection(a_tokens)
            if len(overlap) == 0:
                zero_overlap_count += 1
        query_types.add(q.get("query_type"))
                
    status = "PASS"
    if num_queries != 40: status = f"FAIL (Expected 40 queries, got {num_queries})"
    if empty_queries > 0: status = "FAIL (Empty queries found)"
    if invalid_answer_ids > 0: status = "FAIL (Invalid answer IDs)"
    if duplicates > 0: status = "FAIL (Duplicate queries found)"
    if zero_overlap_count < 8: status = f"FAIL (Zero overlap count < 8: {zero_overlap_count})"
    if not {"semantic", "attributed", "temporal"}.issubset(query_types):
        status = "FAIL (Missing a required query shape)"
    
    print("\n## EVALUATION SET VALIDATION")
    print("-" * 25)
    print(f"Queries: {num_queries}")
    print(f"Valid answer IDs: {num_queries - invalid_answer_ids}/{num_queries}")
    print(f"Duplicates: {duplicates}")
    print(f"Zero-overlap queries: {zero_overlap_count}")
    print(f"Required zero-overlap queries: >= 8")
    print(f"Query types: {', '.join(sorted(t for t in query_types if t))}")
    print(f"Status: {status}")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    corpus_filepath = os.path.join(project_root, 'data', 'chat_corpus_2.json')
    queries_filepath = os.path.join(project_root, 'data', 'evaluation', 'test_queries_40.json')
    corpus, valid_ids = validate_corpus(corpus_filepath)
    validate_queries(queries_filepath, corpus, valid_ids)
