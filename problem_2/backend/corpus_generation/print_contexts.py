import json

with open(r'c:\Users\ASUS\OneDrive\Desktop\akhil\data\chat_corpus_2.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

with open('contexts_output.txt', 'w', encoding='utf-8') as out:
    for target_id in [257, 3764, 4304]:
        out.write(f"\n=============================\n")
        out.write(f"Context for Answer ID {target_id}\n")
        out.write(f"=============================\n")
        # Find index of target_id
        idx = next((i for i, m in enumerate(corpus) if m['id'] == target_id), -1)
        if idx == -1:
            continue
        start = max(0, idx - 20)
        end = min(len(corpus), idx + 5)
        for j in range(start, end):
            out.write(f"[{corpus[j]['id']}] {corpus[j]['sender']}: {corpus[j]['text']}\n")
