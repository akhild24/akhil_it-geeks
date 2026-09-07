# Project Requirement Document
## Semantic + Attributed + Temporal Search over Hinglish Group Chat

**Author:** Akhil Dwivedi
**Version:** 1.0
**Date:** September 2026

---

## 1. Problem Statement

Build a search system for a WhatsApp-style group chat corpus that can answer three kinds of queries:

1. **Semantic** — "when did we decide on the trip" (meaning-based, no shared vocabulary with the actual message)
2. **Attributed** — "what did Priya say about budget" (person-first filtering)
3. **Temporal** — "what did we discuss last month" (date-first filtering)

The system must handle Hinglish (Hindi + English code-mixed) text, informal chat style (typos, "haan," "lol," short replies), and must correctly say **no results found** when nothing relevant exists, rather than hallucinating a match.

---

## 2. Goals and Non-Goals

### Goals
- Build a working end-to-end search pipeline: corpus → embeddings → retrieval → context → UI
- Correctly answer queries where the query shares **zero surface words** with the answer message (minimum 8 such test cases required)
- Correctly classify and route attributed vs temporal vs semantic queries
- Return matched message with surrounding conversational context (not an isolated line)
- Demonstrate retrieval quality with a measurable evaluation set (not just "it looks right")

### Non-Goals (out of scope for v1)
- Multi-group / multi-chat support (single corpus only)
- Real-time ingestion of live chats
- Voice or image message search
- Authentication / multi-user accounts
- Production-grade scaling (this is an assignment-scale system, not a startup)

---

## 3. Core Architectural Decision (Important Revision)

> **Do not rely on pure embedding similarity search.** This is the central risk in the project and the most likely reason to fail the "zero word overlap" test cases.

### Why pure embeddings fail here
A message like *"chalo fix hai, 14 tareek ko nikalte hain"* has no strong semantic neighbor to *"decide"* or *"trip"* in many multilingual sentence embedding spaces, especially when the message is short, code-mixed, and colloquial. Pure cosine similarity search over such short noisy text is unreliable in exactly the cases the assignment is designed to test.

### Required approach: Hybrid Retrieval
Combine **three signals**, not one:

| Layer | Purpose | Tool |
|---|---|---|
| 1. Keyword/lexical search | Catches exact-term and attributed queries, cheap and precise | `rank_bm25` (BM25Okapi) |
| 2. Dense embedding search | Catches semantic/meaning-based queries | Multilingual sentence embeddings |
| 3. Fusion / reranking | Combines both signal sets into one ranked list | Reciprocal Rank Fusion (RRF) or weighted score fusion |
| 4. Metadata filter | Pre-filters candidate pool for attributed/temporal queries | Sender + timestamp fields |

**Critical model choice:** Use `intfloat/multilingual-e5-large` or `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for embeddings — **not** `all-MiniLM-L6-v2`. MiniLM-L6 is English-trained; it will not throw an error on Hinglish, it will simply give subtly worse similarity scores, and you likely won't notice until your hardest test cases silently fail.

---

## 4. System Architecture

```
                         ┌─────────────────────┐
                         │   User Query Input    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Query Classifier     │
                         │ (semantic / attributed │
                         │  / temporal / mixed)   │
                         └──────────┬───────────┘
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │ Attributed Path    │ │ Temporal Path      │ │ Semantic Path      │
    │ filter by sender   │ │ filter by date     │ │ full corpus search │
    │ then search inside │ │ then search inside │ │                    │
    └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
              └─────────────────────┼─────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Hybrid Retrieval Engine        │
                    │  BM25 (lexical) + Dense (embed)  │
                    │      → Fusion / Rerank            │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Context Window Builder          │
                    │   (±5 messages around match)      │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   FastAPI /search endpoint        │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Frontend (React or HTML)        │
                    │  search bar + filters + results   │
                    └───────────────────────────────┘
```

---

## 5. Functional Requirements

### FR1 — Corpus Generation
- Generate 4,000+ synthetic Hinglish messages via Gemini API
- 8 distinct participants with consistent personas/writing styles
- 6-month time span with realistic date/time distribution (not uniform — bursts, gaps, late-night chats)
- Include natural noise: typos, abbreviations ("kl", "ok", "lol"), short one-word replies, emoji-only messages
- Output format: structured JSON — `{id, sender, timestamp, text, prev_id, next_id}`

### FR2 — Indexing Pipeline
- Compute dense embeddings for every message using multilingual model
- Build BM25 index over tokenized message text
- Store embeddings, BM25 index, and metadata (sender, timestamp) in a persistent store (JSON files or MongoDB)
- Store `prev_id` / `next_id` (or flat array index) for O(1) context window retrieval

### FR3 — Query Classifier
- Detects query type using simple rule-based signals first (fast, explainable), falling back to an LLM call for ambiguous cases:
  - Attributed cues: named participant present in query
  - Temporal cues: date/time phrases ("last month," "in March," "yesterday")
  - Otherwise: semantic
- Must support **mixed queries** ("what did Priya say last month about the trip") — apply both filters, then semantic-rank within the filtered set

### FR4 — Hybrid Search Execution
- Run BM25 and dense search in parallel over the (possibly pre-filtered) candidate set
- Fuse rankings (Reciprocal Rank Fusion recommended — simple, no weight-tuning required)
- Return top-K (e.g. top 5) fused results

### FR5 — Context Window
- For each returned hit, fetch ±5 surrounding messages
- Highlight the matched message distinctly from its context in the response payload

### FR6 — "Not Covered" Handling
- If fused top score falls below a calibrated threshold, return "no relevant messages found" rather than a weak forced match
- This threshold should be tuned against the test query set (see Section 7)

### FR7 — API
- `POST /search` — body: `{query: str, sender_filter?: str, date_range?: [start, end]}`
- Response: `{query_type, results: [{message, context_before, context_after, score}], no_match: bool}`

### FR8 — Frontend
- Search bar
- Optional sender dropdown filter
- Optional date range picker
- Result cards showing matched message highlighted within surrounding context
- Empty state for no-match queries

---

## 6. Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Corpus generation | Gemini API | Prompt per-persona for style consistency |
| Embeddings | `multilingual-e5-large` or `paraphrase-multilingual-mpnet-base-v2` | Do not use English-only MiniLM |
| Lexical search | `rank_bm25` | Lightweight, no external service needed |
| Fusion | Reciprocal Rank Fusion (custom, ~20 lines) | Avoids weight-tuning headaches of linear fusion |
| Vector similarity | NumPy cosine similarity, or FAISS if corpus grows | FAISS optional at this scale (4K messages is small) |
| Backend | FastAPI | |
| Frontend | React (or plain HTML/JS if time-constrained) | |
| Storage | JSON files or MongoDB | MongoDB preferred if you want filter queries to be fast/native |
| Query classification | Rule-based + Gemini API fallback | |

---

## 7. Evaluation Plan

This is the part graders/reviewers will actually scrutinize — build it deliberately, not as an afterthought.

- **40 test queries minimum**, spanning all three types
- **At least 8 zero-overlap queries**: query shares no words with the target message, forces the system to rely on real semantic/hybrid retrieval rather than lucky keyword overlap
- **A handful of "should return nothing" queries**: things never discussed in the corpus, to test the not-covered path
- Metrics to report:
  - Precision@1 / Precision@5
  - Manual pass/fail per test query with the specific correct message ID
  - Separate breakdown by query type (semantic / attributed / temporal / mixed) — this shows which component is actually doing the work

---

## 8. What You Already Know vs What's New

| Already know (from resume) | New for this project |
|---|---|
| FastAPI, React, MongoDB | Sentence Transformers library (~1-2 hrs to learn) |
| RAG pipeline experience (47Billion internship) | BM25 / lexical search basics (~1 hr) |
| Gemini API usage | Reciprocal Rank Fusion (~30 min, simple to implement) |
| Multilingual/Hinglish handling (Udaan project) | Retrieval evaluation methodology (precision@k) |

Net new learning is small — this is a good scope match for your background.

---

## 9. Suggested Timeline

| Phase | Task | Est. Time |
|---|---|---|
| 1 | Corpus generation + cleaning | 1 day |
| 2 | Indexing pipeline (BM25 + embeddings) | 1 day |
| 3 | Query classifier + fusion logic | 1–1.5 days |
| 4 | FastAPI backend + endpoints | 1 day |
| 5 | Frontend | 1 day |
| 6 | Test query set + evaluation + threshold tuning | 1 day |
| 7 | Polish, edge cases, README | 0.5 day |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Pure embedding search fails zero-overlap test cases | Hybrid BM25 + embedding + fusion (Section 3) — non-negotiable |
| Wrong embedding model silently underperforms on Hinglish | Use multilingual-e5-large or mpnet, verify manually on 5–10 Hinglish pairs before building the rest |
| Synthetic corpus feels too clean/uniform | Explicitly prompt Gemini for typos, mixed message lengths, bursty timing, one-word replies |
| Classifier misroutes mixed queries | Design classifier to apply filters additively rather than picking one exclusive path |
| No-match threshold miscalibrated (too strict or too loose) | Tune against evaluation set explicitly, don't guess a static number |

---

## 11. Scope for Improvement (Beyond Baseline)

If time permits after the core system works, these add real depth without huge extra effort:

1. **Cross-encoder reranker** (e.g. `bge-reranker-base`) on top of the fused top-20 — meaningfully improves precision@1, cheap to add since it's just a rescoring pass.
2. **Conversation thread detection** — group replies/reactions so context windows follow a topic thread rather than strict chronological order.
3. **Query rewriting via LLM** — expand vague queries ("that trip thing") into more search-friendly forms before retrieval.
4. **Confidence-calibrated no-match** — instead of a fixed threshold, train a tiny logistic regression on (BM25 score, dense score, score gap) → match/no-match, using your 40 labeled queries as training data. More defensible than a magic number.
5. **Analytics dashboard** — show per-query-type accuracy live, useful both as a demo feature and as a self-check tool.
6. **Multilingual query support** — allow queries typed in Devanagari script as well as Roman Hinglish, testing whether the embedding model bridges scripts (a genuinely interesting stretch goal given the multilingual model choice).
7. **FAISS/IVF index** — swap NumPy brute-force cosine for FAISS if you want to demonstrate scalability awareness, even though 4K messages doesn't strictly need it.

---

## 12. Deliverables Checklist

- [ ] Synthetic corpus (JSON, 4,000+ messages, 8 participants, 6 months)
- [ ] Indexing pipeline (BM25 + dense embeddings)
- [ ] Query classifier
- [ ] Hybrid retrieval + fusion logic
- [ ] Context window builder
- [ ] FastAPI backend with `/search` endpoint
- [ ] Frontend UI
- [ ] 40-query evaluation set with results and precision@k report
- [ ] README documenting architecture and design decisions
