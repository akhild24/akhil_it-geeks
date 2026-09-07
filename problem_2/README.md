# Semantic + Attributed + Temporal Search over Hinglish Group Chat

A search engine for WhatsApp-style Hinglish group chats built using Hybrid Search (BM25 + Dense Embeddings) and Reciprocal Rank Fusion.

## Project Structure
- `backend/`: FastAPI application
  - `corpus_generation/`: Scripts to generate Hinglish synthetic chat data.
  - `indexing/`: Logic for dense embeddings and BM25 tokenization.
  - `retrieval/`: Hybrid search pipeline and fusion algorithms.
  - `query_classification/`: Rule-based and LLM fallback classifiers.
  - `evaluation/`: Automated evaluation pipeline on the test queries.
- `frontend/`: React + Vite application
- `data/`: Storage for the corpus, embeddings, and indices.

## Setup Instructions

### Backend
1. `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. Build Indexes: `python -m backend.indexing.build_indexes` 
   *(Note: This uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for dense embeddings and `BM25Okapi` with a custom Hinglish tokenizer for lexical search. Indexes are persisted to `data/indexes/`)*
5. Start server: `cd backend` and `uvicorn main:app --reload`

### Phase 4 Validation
To run the Phase 4 hybrid retrieval and query classification tests:
`python -m backend.evaluation.phase4_validation`

## Architecture Highlights
- **Query Classification:** Determines if a query is `semantic`, `attributed`, `temporal`, or `mixed` using deterministic logic.
- **Entity Extraction:** Matches sender names to known participants using boundary-aware regex.
- **Temporal Parsing:** Extracts dates and date ranges based on relative or explicit temporal cues, mapped against a deterministic reference date (`2026-07-11`).
- **Metadata Filtering:** Reduces the candidate pool by intersecting sender and temporal constraints.
- **Hybrid Retrieval:** Runs BM25 and Dense vector retrieval *only* on the filtered candidates.
- **Reciprocal Rank Fusion (RRF):** Fuses lexical and semantic scores using `1 / (k + rank)` against the message ID, balancing results deterministically.

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`
