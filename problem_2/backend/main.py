from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
from datetime import datetime

from backend.query_classification.classifier import classify_query
from backend.retrieval.hybrid_retrieval import hybrid_search
from backend.retrieval.context_builder import build_context

app = FastAPI(title="Semantic Group Chat Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only, should be specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_SENDERS = {"priya", "meera", "aditya", "neha", "simran", "kunal", "akhil", "rohan"}

# Pydantic Models
class SearchRequest(BaseModel):
    query: str
    sender: Optional[str] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    top_k: Optional[int] = 5

class ContextMessage(BaseModel):
    id: int
    sender: str
    timestamp: str
    text: str
    is_match: bool

class SearchResult(BaseModel):
    message_id: int
    sender: str
    timestamp: str
    text: str
    fused_score: float
    fused_rank: int
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    highlight_terms: List[str] = []
    context: List[ContextMessage]

class FilterMetadata(BaseModel):
    sender: Optional[str] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None

class SearchResponse(BaseModel):
    query: str
    query_type: str
    filters: FilterMetadata
    results: List[SearchResult]
    no_match: bool  # Currently only True if candidate set is empty, as RRF is not a reliable confidence score.


def highlight_terms(query: str, text: str) -> List[str]:
    """Return query terms that occur in a result without exposing HTML markup."""
    query_terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 1}
    text_terms = {term.lower() for term in re.findall(r"\w+", text)}
    return sorted(query_terms.intersection(text_terms), key=lambda term: (-len(term), term))

# Phase 9: No-match detection is now handled by a confidence threshold in hybrid_search.
# The strategy uses a combined score (top_dense_raw + top5_overlap * 0.05) to evaluate confidence.
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/")
def read_root():
    return {"message": "Welcome to Semantic Group Chat Search API"}

@app.post("/search", response_model=SearchResponse)
def search_api(request: SearchRequest):
    query = request.query.strip()
    
    # 1. Classify query
    classification = classify_query(query)
    query_type = classification.query_type
    
    # 2. Filter Precedence (Explicit > Extracted)
    final_sender = request.sender if request.sender is not None else classification.sender
    
    if request.date_start and request.date_end:
        if request.date_start > request.date_end:
            raise HTTPException(status_code=400, detail="date_start cannot be after date_end")
        final_date_start = request.date_start
        final_date_end = request.date_end
    elif request.date_start or request.date_end:
        final_date_start = request.date_start if request.date_start is not None else classification.date_start
        final_date_end = request.date_end if request.date_end is not None else classification.date_end
    else:
        final_date_start = classification.date_start
        final_date_end = classification.date_end
        
    if final_sender and final_sender.lower() not in VALID_SENDERS:
        raise HTTPException(status_code=400, detail=f"Invalid sender: {final_sender}")
        
    if request.top_k and request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")
        
    if not query:
        return SearchResponse(
            query=query,
            query_type="semantic",
            filters=FilterMetadata(sender=final_sender, date_start=final_date_start, date_end=final_date_end),
            results=[],
            no_match=True
        )
        
    # 3. Retrieval
    results = hybrid_search(
        query=query, 
        sender=final_sender, 
        date_start=final_date_start, 
        date_end=final_date_end, 
        top_k=request.top_k or 5
    )
    
    # 4. No-match decision
    # Phase 9 confidence threshold rejects queries returning an empty list.
    # We return no_match=True if hybrid_search returns an empty candidate set.
    if not results:
        return SearchResponse(
            query=query,
            query_type=query_type,
            filters=FilterMetadata(sender=final_sender, date_start=final_date_start, date_end=final_date_end),
            results=[],
            no_match=True
        )
        
    # 5. Build response
    api_results = []
    for r in results:
        context_msgs = build_context(r['message_id'])
        api_results.append(SearchResult(
            message_id=r['message_id'],
            sender=r['metadata']['sender'],
            timestamp=r['metadata']['timestamp'],
            text=r['message'],
            fused_score=r['fused_score'],
            fused_rank=r['fused_rank'],
            bm25_rank=r.get('bm25_rank'),
            dense_rank=r.get('dense_rank'),
            highlight_terms=highlight_terms(query, r['message']),
            context=context_msgs
        ))
        
    return SearchResponse(
        query=query,
        query_type=query_type,
        filters=FilterMetadata(sender=final_sender, date_start=final_date_start, date_end=final_date_end),
        results=api_results,
        no_match=False
    )
