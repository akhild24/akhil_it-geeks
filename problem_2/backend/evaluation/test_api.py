import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Backend is running"}

def test_a_semantic_search():
    res = client.post("/search", json={"query": "What did we decide about the trip?"})
    assert res.status_code == 200
    data = res.json()
    assert data["query_type"] == "semantic"
    assert data["no_match"] is False
    assert len(data["results"]) > 0

def test_a1_manali_decision_search():
    res = client.post("/search", json={"query": "When did we decide on Manali?", "top_k": 10})
    assert res.status_code == 200
    data = res.json()
    assert data["query_type"] == "semantic"
    assert any(result["message_id"] == 2863 for result in data["results"])

def test_a2_prd_filter_aliases():
    res = client.post("/search", json={
        "query": "What did we discuss?",
        "sender_filter": "Priya",
        "date_range": ["2026-05-01T00:00:00", "2026-05-31T23:59:59"]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["filters"]["sender"] == "Priya"
    assert data["filters"]["date_start"] is not None

def test_b_attributed_search():
    res = client.post("/search", json={"query": "What did Priya say about budget?"})
    assert res.status_code == 200
    data = res.json()
    assert data["query_type"] == "attributed"
    assert data["filters"]["sender"] == "Priya"
    assert data["no_match"] is False

def test_c_temporal_search():
    res = client.post("/search", json={"query": "What did we discuss in June 2026?"})
    assert res.status_code == 200
    data = res.json()
    assert data["query_type"] == "temporal"
    assert data["filters"]["date_start"] is not None

def test_d_mixed_search():
    # Semantic query with explicit API filters acting as mixed search
    res = client.post("/search", json={
        "query": "What did we discuss?",
        "sender": "Aditya",
        "date_start": "2026-06-01T00:00:00",
        "date_end": "2026-06-30T23:59:59"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["filters"]["sender"] == "Aditya"
    assert data["filters"]["date_start"] == "2026-06-01T00:00:00Z" or "2026-06-01T00:00:00" in data["filters"]["date_start"]

def test_e_explicit_sender_filter():
    res = client.post("/search", json={"query": "hello", "sender": "Priya"})
    assert res.status_code == 200
    for r in res.json()["results"]:
        assert r["sender"].lower() == "priya"

def test_f_explicit_date_filter():
    res = client.post("/search", json={"query": "hello", "date_start": "2026-06-01T00:00:00", "date_end": "2026-06-02T00:00:00"})
    assert res.status_code == 200

def test_g_explicit_sender_and_date():
    res = client.post("/search", json={"query": "hello", "sender": "Neha", "date_start": "2026-06-01T00:00:00", "date_end": "2026-06-02T00:00:00"})
    assert res.status_code == 200
    for r in res.json()["results"]:
        assert r["sender"].lower() == "neha"

def test_h_no_match():
    # Phase 9: The system now correctly identifies genuine no-match queries
    # using a confidence threshold without relying on impossible filters.
    res = client.post("/search", json={"query": "Did anyone figure out the processing time for the Schengen visa?"})
    assert res.status_code == 200
    data = res.json()
    assert data["no_match"] is True
    assert len(data["results"]) == 0

def test_i_query_31():
    res = client.post("/search", json={"query": "What did the group decide about the Manali trip?", "top_k": 10})
    assert res.status_code == 200
    data = res.json()
    assert data["filters"]["date_start"] is None
    assert any(r["message_id"] == 2863 for r in data["results"])

def test_j_empty_query():
    res = client.post("/search", json={"query": ""})
    assert res.status_code == 200
    assert res.json()["no_match"] is True

def test_k_whitespace_query():
    res = client.post("/search", json={"query": "   "})
    assert res.status_code == 200
    assert res.json()["no_match"] is True

def test_l_invalid_sender():
    res = client.post("/search", json={"query": "hello", "sender": "invalid_user"})
    assert res.status_code == 400

def test_m_invalid_date():
    res = client.post("/search", json={"query": "hello", "date_start": "not-a-date"})
    assert res.status_code == 422 # Pydantic validation error

def test_n_invalid_date_range():
    res = client.post("/search", json={
        "query": "hello", 
        "date_start": "2026-06-02T00:00:00", 
        "date_end": "2026-06-01T00:00:00"
    })
    assert res.status_code == 400

def test_o_invalid_top_k():
    res = client.post("/search", json={"query": "hello", "top_k": -1})
    assert res.status_code == 400

def test_p_valid_top_k():
    res = client.post("/search", json={"query": "What did we decide about the trip?", "top_k": 2})
    assert res.status_code == 200
    assert len(res.json()["results"]) <= 2
