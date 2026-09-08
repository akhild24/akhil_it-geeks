import os
import json
import numpy as np

from backend.indexing.embedding_index import EmbeddingIndexer
from backend.indexing.bm25_index import BM25Indexer
from backend.indexing.index_utils import save_embeddings, save_bm25, save_metadata, INDEXES_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_PATH = os.path.join(BASE_DIR, 'data', 'chat_corpus_2.json')

def main():
    print("Loading Corpus 2...")
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus not found at {CORPUS_PATH}")
        return

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Validate basic schema
    if not all(isinstance(msg, dict) and 'id' in msg and 'text' in msg for msg in corpus):
        print("Error: Corpus validation failed. Each message must be a dictionary with 'id' and 'text'.")
        return

    num_messages = len(corpus)
    print(f"Loaded {num_messages} messages.")

    # Prepare data for indexing
    # Contextual indexing keeps search focused on the meaning of a chat turn.
    # The visible result remains the original message; the enrichment is only
    # used by BM25 and the multilingual embedding model.
    documents = [
        f"{msg.get('sender', '')}: {msg['text']}\n{msg.get('search_context', '')}".strip()
        for msg in corpus
    ]
    metadata = []
    for idx, msg in enumerate(corpus):
        # We store enough metadata to map vectors back to: id, sender, timestamp, text, corpus position
        metadata.append({
            'corpus_index': idx,
            'id': msg.get('id'),
            'sender': msg.get('sender'),
            'timestamp': msg.get('timestamp'),
            'text': msg.get('text'),
            'search_context': msg.get('search_context', ''),
        })

    # 1. Build Dense Embeddings
    print("\nInitializing Embedding Model...")
    emb_indexer = EmbeddingIndexer()
    print("Generating dense embeddings...")
    embeddings = emb_indexer.encode_documents(documents)
    
    if len(embeddings) != num_messages:
        print(f"Error: Embedding count ({len(embeddings)}) does not match corpus count ({num_messages}).")
        return
        
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("Error: NaN or Inf values found in embeddings.")
        return

    # 2. Build BM25 Index
    print("\nBuilding BM25 Index...")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build(documents)
    
    # Validation check for BM25 docs count
    bm25_doc_count = len(bm25_indexer.tokenized_corpus)
    if bm25_doc_count != num_messages:
        print(f"Error: BM25 documents count ({bm25_doc_count}) does not match corpus count ({num_messages}).")
        return

    # 3. Save Indexes and Metadata
    print("\nSaving indexes to disk...")
    save_embeddings(embeddings)
    save_bm25(bm25_indexer.bm25)
    save_metadata(metadata)

    # 4. Print Summary
    print("\n==================================================")
    print("INDEX BUILD SUMMARY")
    print("==================================================")
    print(f"Corpus: chat_corpus_2.json")
    print(f"Messages: {num_messages}")
    print(f"Embedding model: {emb_indexer.model_name}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"BM25 documents: {bm25_doc_count}")
    print(f"Indexes saved: {INDEXES_DIR}")
    print(f"Status: PASS")

if __name__ == "__main__":
    main()
