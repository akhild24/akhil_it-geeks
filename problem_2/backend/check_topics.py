import json
import re

with open(r'c:\Users\ASUS\OneDrive\Desktop\akhil\data\chat_corpus_2.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

topics = {
    "gardening": ["gardening", "fertilizer", "succulent", "watering plants", "balcony garden"],
    "skiing": ["skiing", "snowboard", "ice skating", "ski resort", "snow gear"],
    "astronomy": ["astronomy", "telescope", "stargazing", "meteor shower", "planetarium"],
    "scuba": ["scuba diving", "snorkeling", "coral reef", "wetsuit"],
    "baking": ["baking bread", "sourdough", "kneading", "yeast"],
    "renovation": ["home renovation", "plumbing", "contractor", "repainting walls"],
    "vr": ["vr headset", "virtual reality", "oculus", "meta quest"],
    "magic": ["magic tricks", "illusion", "magician", "card trick"],
    "knitting": ["knitting", "crochet", "yarn", "sewing machine"]
}

for topic, keywords in topics.items():
    hits = 0
    for msg in corpus:
        text = str(msg.get('text', '')).lower()
        if any(k in text for k in keywords):
            hits += 1
    print(f"Topic '{topic}': {hits} hits")
