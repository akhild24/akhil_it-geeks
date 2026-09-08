import { useState, useEffect, useCallback } from 'react'
import './App.css'
import SearchBar from './components/SearchBar'
import ResultCard from './components/ResultCard'
import QueryMeta from './components/QueryMeta'
import { searchMessages, checkHealth } from './api/search'

const EXAMPLE_QUERIES = [
  "What did we decide about the trip?",
  "What did Priya say about budget?",
  "What did we discuss in June?",
  "What time had been fixed for the Saturday plan?",
]

function App() {
  // Health check
  const [health, setHealth] = useState('loading') // 'loading' | 'ok' | 'error'

  // Search state
  const [isLoading, setIsLoading] = useState(false)
  const [searchResponse, setSearchResponse] = useState(null)
  const [error, setError] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)

  // Ref to allow SearchBar to set query from example clicks
  const [exampleQuery, setExampleQuery] = useState(null)

  useEffect(() => {
    checkHealth()
      .then(() => setHealth('ok'))
      .catch(() => setHealth('error'))
  }, [])

  const handleSearch = useCallback(async ({ query, sender, dateStart, dateEnd }) => {
    setIsLoading(true)
    setError(null)
    setSearchResponse(null)
    setHasSearched(true)

    try {
      const data = await searchMessages({ query, sender, dateStart, dateEnd, topK: 10 })
      setSearchResponse(data)
    } catch (err) {
      console.error('[RecallAI] Search error:', err)
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleExampleClick = (q) => {
    setExampleQuery(q)
  }

  // Determine what to render in the results area
  const renderContent = () => {
    // Loading
    if (isLoading) {
      return (
        <div className="loading-state" id="loading-state">
          <div className="loading-spinner-lg" />
          <p>Searching conversations…</p>
        </div>
      )
    }

    // Error
    if (error) {
      return (
        <div className="error-state" id="error-state">
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <h3>Unable to connect to RecallAI search service</h3>
          <p>
            {error.includes('Failed to fetch') || error.includes('NetworkError')
              ? 'Make sure the backend is running on port 8000.'
              : error
            }
          </p>
          <button
            className="retry-btn"
            onClick={() => { setError(null); setHasSearched(false); }}
            id="retry-button"
          >
            Try again
          </button>
        </div>
      )
    }

    // No-match
    if (searchResponse && searchResponse.no_match) {
      return (
        <div className="results-section">
          <QueryMeta queryType={searchResponse.query_type} filters={searchResponse.filters} />
          <div className="no-match-state" id="no-match-state">
            <svg className="no-match-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
              <line x1="8" y1="8" x2="14" y2="14" />
              <line x1="14" y1="8" x2="8" y2="14" />
            </svg>
            <h3>No matching conversation found</h3>
            <p>Try using different keywords, a different date range, or removing a filter.</p>
          </div>
        </div>
      )
    }

    // Results
    if (searchResponse && searchResponse.results && searchResponse.results.length > 0) {
      return (
        <div className="results-section" id="results-section">
          <QueryMeta queryType={searchResponse.query_type} filters={searchResponse.filters} />
          <div className="results-header">
            <span className="results-count">
              {searchResponse.results.length} result{searchResponse.results.length !== 1 ? 's' : ''} found
            </span>
          </div>
          <div className="results-list" id="results-list">
            {searchResponse.results.map((result, i) => (
              <ResultCard key={result.message_id} result={result} rank={i + 1} />
            ))}
          </div>
        </div>
      )
    }

    // Welcome / initial state
    if (!hasSearched) {
      return (
        <div className="welcome-state" id="welcome-state">
          <svg className="welcome-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <h2>Search your conversations</h2>
          <p>
            Ask questions in natural language to find messages, decisions, and discussions across your group chat.
          </p>
          <div className="example-queries">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                className="example-query"
                onClick={() => handleExampleClick(q)}
                type="button"
              >
                "{q}"
              </button>
            ))}
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </div>
          <h1 className="app-title">RecallAI</h1>
        </div>
        <p className="app-subtitle">
          <span className={`health-dot ${health}`} title={`Backend: ${health}`} />
          Semantic conversation search
        </p>
      </header>

      <main>
        <SearchBar
          onSearch={handleSearch}
          isLoading={isLoading}
          exampleQuery={exampleQuery}
          onExampleConsumed={() => setExampleQuery(null)}
        />
        {renderContent()}
      </main>

      <footer className="app-footer">
        RecallAI · Semantic + Attributed + Temporal Search
      </footer>
    </div>
  )
}

export default App
