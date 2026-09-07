import json
with open(r'c:\Users\ASUS\OneDrive\Desktop\akhil\data\chat_corpus_2.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)
seen = set()
for i, m in enumerate(corpus):
    if not m.get('id'):
        print(f'Missing ID at index {i}')
    elif m['id'] in seen:
        print(f'Duplicate ID at index {i}: {m["id"]}')
    else:
        seen.add(m['id'])
    
    if i > 0 and m.get('prev_id') != corpus[i-1]['id']:
        print(f'Link mismatch at index {i}: prev_id={m.get("prev_id")}, actual_prev={corpus[i-1]["id"]}')
