import json
import re

with open(r'c:\Users\ASUS\OneDrive\Desktop\akhil\data\chat_corpus_2.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Find keywords
keywords = ['decided', 'decision', 'finalized', "let's go with", "agreed", "final decision", "conclusion"]

results = []
for i, msg in enumerate(corpus):
    text = msg['text'].lower()
    if any(k in text for k in keywords):
        results.append((i, msg))

with open('decisions_output.txt', 'w', encoding='utf-8') as out:
    for idx, msg in results:
        start = max(0, idx - 10)
        end = min(len(corpus), idx + 5)
        out.write(f"\n--- Decision around ID {msg['id']} ---\n")
        for j in range(start, end):
            out.write(f"[{corpus[j]['id']}] {corpus[j]['sender']}: {corpus[j]['text']}\n")

