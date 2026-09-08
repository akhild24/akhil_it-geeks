import os
import json
import re
from datetime import datetime
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND = os.path.join(ROOT, "backend")
DATA = os.path.join(ROOT, "data")
FRONTEND = os.path.join(ROOT, "frontend")

def check_file(path):
    return os.path.isfile(path)

def check_dir(path):
    return os.path.isdir(path)

print("==================================================")
print("PART A — PROJECT FOUNDATION AUDIT")
print("=================================")

# 1. backend/main.py
main_py = os.path.join(BACKEND, "main.py")
if check_file(main_py):
    with open(main_py, 'r', encoding='utf-8') as f:
        content = f.read()
    fastapi_exists = "FastAPI" in content
    health_endpoint = "/health" in content
    cors_config = "CORSMiddleware" in content
    print(f"backend/main.py exists: PASS")
    print(f"  FastAPI app exists: {'PASS' if fastapi_exists else 'FAIL'}")
    print(f"  /health endpoint exists: {'PASS' if health_endpoint else 'FAIL'}")
    print(f"  CORS config exists: {'PASS' if cors_config else 'FAIL'}")
else:
    print("backend/main.py exists: FAIL")

# 2. backend/config.py
config_py = os.path.join(BACKEND, "config.py")
if check_file(config_py):
    try:
        with open(config_py, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"backend/config.py exists: PASS")
    except Exception as e:
        print(f"backend/config.py read error: {e}")
else:
    print("backend/config.py exists: FAIL")

# 3. requirements.txt
req_txt = os.path.join(ROOT, "requirements.txt")
if check_file(req_txt):
    print("requirements.txt exists: PASS")
else:
    print("requirements.txt exists: FAIL")

# 4. frontend
print(f"frontend exists: {'PASS' if check_dir(FRONTEND) else 'FAIL'}")

# 5. Project directories
dirs = [
    "backend/corpus_generation",
    "backend/indexing",
    "backend/retrieval",
    "backend/query_classification",
    "backend/evaluation",
    "data"
]
for d in dirs:
    print(f"Directory {d}: {'PASS' if check_dir(os.path.join(ROOT, d)) else 'FAIL'}")

print("\n==================================================")
print("PART B & C — CORPUS AUDITS")
print("=======================")

def audit_corpus(filename):
    path = os.path.join(DATA, filename)
    print(f"\nAuditing {filename}:")
    if not check_file(path):
        print("FAIL (File not found)")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    count = len(corpus)
    senders = set()
    ids = set()
    missing_fields = 0
    duplicate_ids = 0
    broken_links = 0
    invalid_timestamps = 0
    non_chrono = 0
    future_dates = 0
    empty_texts = 0
    
    start_date = None
    end_date = None
    now = datetime.utcnow().replace(tzinfo=None)
    
    for i, msg in enumerate(corpus):
        if not all(k in msg for k in ['id', 'sender', 'timestamp', 'text', 'prev_id', 'next_id']):
            missing_fields += 1
            
        m_id = msg.get('id')
        if m_id is not None:
            if m_id in ids: duplicate_ids += 1
            ids.add(m_id)
            
        text = msg.get('text', '')
        if not text or not str(text).strip():
            empty_texts += 1
            
        senders.add(msg.get('sender'))
        
        try:
            ts = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
            if start_date is None or ts < start_date: start_date = ts
            if end_date is None or ts > end_date: end_date = ts
            if ts.replace(tzinfo=None) > now: future_dates += 1
            if i > 0:
                prev_ts = datetime.fromisoformat(corpus[i-1]['timestamp'].replace('Z', '+00:00'))
                if ts < prev_ts:
                    non_chrono += 1
        except:
            invalid_timestamps += 1

    # check links
    for i, msg in enumerate(corpus):
        m_id = msg.get('id')
        p_id = msg.get('prev_id')
        n_id = msg.get('next_id')
        
        if p_id is not None and p_id not in ids: broken_links += 1
        if n_id is not None and n_id not in ids: broken_links += 1
        
        if i > 0 and p_id != corpus[i-1].get('id'): broken_links += 1
        if i < count - 1 and n_id != corpus[i+1].get('id'): broken_links += 1

    duration = (end_date - start_date).days if end_date and start_date else 0
    
    print(f"Message count: {count} ({'PASS' if count >= 4000 else 'FAIL'})")
    print(f"Participant count: {len(senders)} ({'PASS' if len(senders)==8 else 'FAIL'})")
    print(f"Participants: {list(senders)}")
    print(f"Missing fields: {missing_fields} ({'PASS' if missing_fields==0 else 'FAIL'})")
    print(f"Duplicate IDs: {duplicate_ids} ({'PASS' if duplicate_ids==0 else 'FAIL'})")
    print(f"Broken links: {broken_links} ({'PASS' if broken_links==0 else 'FAIL'})")
    print(f"Invalid timestamps: {invalid_timestamps} ({'PASS' if invalid_timestamps==0 else 'FAIL'})")
    print(f"Non-chronological dates: {non_chrono} ({'PASS' if non_chrono==0 else 'FAIL'})")
    print(f"Future dates: {future_dates} ({'PASS' if future_dates==0 else 'FAIL'})")
    print(f"Empty texts: {empty_texts} ({'PASS' if empty_texts==0 else 'FAIL'})")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Duration: {duration} days ({'PASS' if 150 <= duration <= 210 else 'FAIL'})")
    return corpus

c1 = audit_corpus("chat_corpus.json")
c2 = audit_corpus("chat_corpus_2.json")

print("\n==================================================")
print("PART D, E, F — EVALUATION AUDIT")
print("===============================")

eval_path = os.path.join(DATA, "evaluation", "test_queries_40.json")
if not check_file(eval_path):
    print("test_queries_40.json exists: FAIL")
else:
    with open(eval_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"Total queries: {len(queries)} ({'PASS' if len(queries)==40 else 'FAIL'})")
    
    empty_queries = sum(1 for q in queries if not q.get('query') or not str(q['query']).strip())
    print(f"Empty queries: {empty_queries} ({'PASS' if empty_queries==0 else 'FAIL'})")
    
    corpus_2_ids = {m['id']: m['text'] for m in c2} if c2 else {}
    invalid_ans = sum(1 for q in queries if q.get('answer_message_id') not in corpus_2_ids and q.get('answer_message_id') is not None)
    print(f"Invalid answer IDs: {invalid_ans} ({'PASS' if invalid_ans==0 else 'FAIL'})")
    
    q_texts = [q.get('query', '') for q in queries]
    duplicates = len(q_texts) - len(set(q_texts))
    print(f"Duplicate queries: {duplicates} ({'PASS' if duplicates==0 else 'FAIL'})")
    
    # Classify queries manually if possible or dump for manual check
    # Let's write them to a file for me to classify
    print("\nQueries written to eval_dump.txt for manual review of types and no-results.")
    with open("eval_dump.txt", "w", encoding="utf-8") as out:
        for i, q in enumerate(queries):
            qt = q.get('query', '')
            aid = q.get('answer_message_id')
            
            # Zero overlap calculation
            def tokenize(text):
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                return set(text.split())
                
            q_tok = tokenize(qt)
            a_tok = tokenize(corpus_2_ids.get(aid, '')) if aid is not None else set()
            overlap = q_tok.intersection(a_tok)
            zero = len(overlap) == 0
            
            out.write(f"[{i+1}] Q: {qt} | A_ID: {aid} | ZeroOverlap: {'YES' if zero else 'NO'} (Shared: {list(overlap)})\n")
