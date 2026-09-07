from sentence_transformers import SentenceTransformer
import time

try:
    print("Loading model...")
    start = time.time()
    # Try the preferred model
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    print(f"Loaded multilingual-e5-large in {time.time() - start:.2f} seconds.")
except Exception as e:
    print(f"Failed to load multilingual-e5-large: {e}")
