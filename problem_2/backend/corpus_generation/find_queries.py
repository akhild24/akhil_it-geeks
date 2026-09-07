import json
with open(r'c:\Users\ASUS\OneDrive\Desktop\akhil\data\evaluation\test_queries_40.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)
for i, q in enumerate(queries):
    text = q['query'].lower()
    if any(word in text for word in ['decision', 'decided', 'agree', 'finalize', 'conclusion', 'final']):
        print(f"Q{i}: {q['query']} -> Ans {q['answer_message_id']}")
