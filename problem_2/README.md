# Semantic + Attributed + Temporal Search over Hinglish Group Chat

A search engine for WhatsApp-style Hinglish group chats built using Hybrid Search (BM25 + Dense Embeddings) and Reciprocal Rank Fusion.

## Project Structure
- `backend/`: FastAPI application
  - `corpus_generation/`: Scripts to generate Hinglish synthetic chat data.
  - `indexing/`: Logic for dense embeddings and BM25 tokenization.
  - `retrieval/`: Hybrid search pipeline and fusion algorithms.
  - `query_classification/`: Deterministic sender and temporal query classification.
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

## Search API (Phase 5)

The `/search` endpoint (POST) allows searching the Hinglish chat corpus with explicit filters, contextual message tracking, and no-match calibration.

### Request Schema

```json
{
  "query": "What did we decide about the trip?",
  "sender": "Priya",           // Optional, overrides query extraction if provided
  "date_start": "2026-06-01T00:00:00", // Optional, overrides query extraction
  "date_end": "2026-06-30T23:59:59",   // Optional, overrides query extraction
  "top_k": 5                   // Optional, defaults to 5
}
```

The API also accepts the PRD field names `sender_filter` and `date_range` (`[start, end]`) for compatibility with the assignment contract.

### Response Schema

```json
{
  "query": "What did we decide about the trip?",
  "query_type": "attributed",
  "filters": {
    "sender": "Priya",
    "date_start": null,
    "date_end": null
  },
  "results": [
    {
      "message_id": 105,
      "sender": "Priya",
      "timestamp": "2026-06-15T14:30:00",
      "text": "Let's stick to the 50k budget for the trip.",
      "fused_score": 0.032,
      "fused_rank": 1,
      "bm25_rank": 2,
      "dense_rank": 1,
      "context": [
         // Array of up to 5 surrounding messages before and after, 
         // with is_match indicating the retrieved target message
      ]
    }
  ],
  "no_match": false
}
```

### Context Builder
Retrieved messages include up to 5 surrounding messages before and after the target message. This utilizes the actual chronological `prev_id` and `next_id` relationships rather than arbitrary array slices, ensuring the context accurately reflects the conversation flow.

### No-Match Detection

RRF is a ranking signal rather than an absolute confidence score, so the API uses a calibrated evidence check in addition to RRF: the highest raw dense similarity plus agreement between BM25 and dense top-five candidates. An empty query, an empty filtered candidate set, or insufficient evidence returns `no_match: true` with no results. The ten out-of-domain queries live in `data/evaluation/no_match_queries.json` and can be checked with `python backend/evaluation/validate_no_match_queries.py`.

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

The frontend runs on `http://localhost:5173` by default (Vite dev server).

The backend URL defaults to `http://127.0.0.1:8000` and can be overridden via the `VITE_API_URL` environment variable.

## Frontend Search UI (Phase 6)

The React frontend provides a polished search interface for querying the Hinglish group chat corpus.

### Features
- **Natural language search**: Users enter queries like "What did Priya say about budget?" and the backend handles query classification (semantic/attributed/temporal/mixed).
- **Sender filter**: Dropdown to filter by participant (Priya, Meera, Aditya, Neha, Simran, Kunal, Akhil, Rohan).
- **Date filters**: Optional start/end date pickers, sent to the backend for temporal filtering.
- **Context display**: Each result shows the matched message + 5 messages before and after, with the match clearly highlighted.
- **Query type badge**: Displays the detected query type (Semantic, Attributed, Temporal, Mixed) as a colored badge.
- **Active filter chips**: Shows active sender/date filters next to the query type badge.
- **Loading state**: Spinner and disabled controls while searching.
- **No-match state**: Clean "No matching conversation found" when `no_match: true`.
- **Error state**: User-friendly error if the backend is unavailable.
- **Empty query validation**: Prevents submission of blank queries.
- **Health check indicator**: Green/red dot showing backend connectivity status.

### Demo Flow
1. Start backend: `cd backend && uvicorn main:app --reload` (with venv activated)
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173`
4. Try example queries from the welcome screen, or type your own:
   - "What did we decide about the trip?"
   - "What did Priya say about budget?" (with Priya sender filter)
   - "What did we discuss in June?" (temporal query)
   - "What time had been fixed for the Saturday plan?" (semantic, message 2956 retrieved)

### Retrieval Notes
The production pipeline always combines BM25 and multilingual dense retrieval with Reciprocal Rank Fusion over each head's top 100 candidates. Match highlighting is token-based and intentionally does not fabricate lexical highlights for semantic-only matches; those remain visibly marked as the retrieved message in their context thread.
