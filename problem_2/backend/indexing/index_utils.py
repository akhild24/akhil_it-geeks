import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Tuple

# We store everything in data/indexes relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEXES_DIR = os.path.join(BASE_DIR, 'data', 'indexes')

def ensure_index_dir():
    if not os.path.exists(INDEXES_DIR):
        os.makedirs(INDEXES_DIR)

def save_embeddings(embeddings: np.ndarray, filename: str = "corpus_2_embeddings.npy"):
    ensure_index_dir()
    path = os.path.join(INDEXES_DIR, filename)
    np.save(path, embeddings)

def load_embeddings(filename: str = "corpus_2_embeddings.npy") -> np.ndarray:
    path = os.path.join(INDEXES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file {path} not found.")
    return np.load(path)

def save_bm25(bm25_obj: Any, filename: str = "corpus_2_bm25.pkl"):
    ensure_index_dir()
    path = os.path.join(INDEXES_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(bm25_obj, f)

def load_bm25(filename: str = "corpus_2_bm25.pkl") -> Any:
    path = os.path.join(INDEXES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 file {path} not found.")
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_metadata(metadata: list, filename: str = "corpus_2_metadata.json"):
    ensure_index_dir()
    path = os.path.join(INDEXES_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(filename: str = "corpus_2_metadata.json") -> list:
    path = os.path.join(INDEXES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file {path} not found.")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
