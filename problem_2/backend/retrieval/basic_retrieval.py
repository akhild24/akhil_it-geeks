import numpy as np
from typing import List, Dict, Any

from backend.indexing.index_utils import load_embeddings, load_bm25, load_metadata
from backend.indexing.embedding_index import EmbeddingIndexer
from backend.indexing.bm25_index import HinglishTokenizer

# Lazy loaded global state
_EMBEDDINGS = None
_METADATA = None
_BM25 = None
_EMBEDDING_INDEXER = None
_TOKENIZER = None

def _ensure_dense_state():
    global _EMBEDDINGS, _METADATA, _EMBEDDING_INDEXER
    if _EMBEDDINGS is None:
        _EMBEDDINGS = load_embeddings()
    if _METADATA is None:
        _METADATA = load_metadata()
    if _EMBEDDING_INDEXER is None:
        _EMBEDDING_INDEXER = EmbeddingIndexer()

def _ensure_bm25_state():
    global _BM25, _METADATA, _TOKENIZER
    if _BM25 is None:
        _BM25 = load_bm25()
    if _METADATA is None:
        _METADATA = load_metadata()
    if _TOKENIZER is None:
        _TOKENIZER = HinglishTokenizer()

def retrieve_dense(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Performs basic dense retrieval using cosine similarity.
    """
    _ensure_dense_state()
    
    # Encode query
    query_vec = _EMBEDDING_INDEXER.encode_query(query)
    
    # Calculate cosine similarity
    # Assumes embeddings are already normalized or computes raw dot product if not.
    # We will compute cosine similarity explicitly to be safe.
    q_norm = np.linalg.norm(query_vec)
    q_vec_norm = query_vec / q_norm if q_norm != 0 else query_vec
    
    doc_norms = np.linalg.norm(_EMBEDDINGS, axis=1, keepdims=True)
    doc_vecs_norm = np.divide(_EMBEDDINGS, doc_norms, out=np.zeros_like(_EMBEDDINGS), where=doc_norms!=0)
    
    scores = np.dot(doc_vecs_norm, q_vec_norm)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            'message': _METADATA[idx]['text'],
            'message_id': _METADATA[idx]['id'],
            'rank': rank + 1,
            'raw_score': float(scores[idx]),
            'metadata': _METADATA[idx]
        })
        
    return results

def retrieve_bm25(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Performs basic lexical retrieval using BM25.
    """
    _ensure_bm25_state()
    
    tokenized_query = _TOKENIZER.tokenize(query)
    scores = _BM25.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            'message': _METADATA[idx]['text'],
            'message_id': _METADATA[idx]['id'],
            'rank': rank + 1,
            'raw_score': float(scores[idx]),
            'metadata': _METADATA[idx]
        })
        
    return results
