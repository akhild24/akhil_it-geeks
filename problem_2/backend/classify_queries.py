import json
import re

with open('c:/Users/ASUS/OneDrive/Desktop/akhil/data/evaluation/test_queries_40.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

for i, q in enumerate(queries):
    qt = q['query'].lower()
    temporal_words = ['when', 'time', 'date', 'hour', 'latest', 'saturday', 'tomorrow', 'afternoon']
    attributed_words = ['who', 'whose', 'priya', 'simran', 'akhil', 'neha', 'meera', 'aditya', 'kunal', 'rohan']
    
    is_temporal = any(w in qt for w in temporal_words)
    is_attributed = any(w in qt for w in attributed_words)
    
    if is_temporal and is_attributed:
        cat = "Mixed"
    elif is_temporal:
        cat = "Temporal"
    elif is_attributed:
        cat = "Attributed"
    else:
        cat = "Semantic"
        
    print(f"Q{i+1}: {q['query']} | Type: {cat}")
