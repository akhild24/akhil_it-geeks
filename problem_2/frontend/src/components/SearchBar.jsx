import { useState, useEffect } from 'react';

const SENDERS = ['Priya', 'Meera', 'Aditya', 'Neha', 'Simran', 'Kunal', 'Akhil', 'Rohan'];

export default function SearchBar({ onSearch, isLoading, exampleQuery, onExampleConsumed }) {
  const [query, setQuery] = useState('');
  const [sender, setSender] = useState('');
  const [dateStart, setDateStart] = useState('');
  const [dateEnd, setDateEnd] = useState('');
  const [validationMsg, setValidationMsg] = useState('');
  const [filtersOpen, setFiltersOpen] = useState(false);

  // When an example query is passed from the parent, populate and submit
  useEffect(() => {
    if (exampleQuery && !isLoading) {
      setQuery(exampleQuery);
      setValidationMsg('');
      onSearch({ query: exampleQuery, sender, dateStart, dateEnd });
      if (onExampleConsumed) onExampleConsumed();
    }
  }, [exampleQuery]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setValidationMsg('Enter a search query first.');
      return;
    }
    setValidationMsg('');
    onSearch({ query: trimmed, sender, dateStart, dateEnd });
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit} id="search-form">
      <div className="search-input-row">
        <div className="search-input-wrap">
          <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            id="search-input"
            type="text"
            placeholder='Search conversations… e.g. "What did Priya say about budget?"'
            value={query}
            onChange={(e) => { setQuery(e.target.value); setValidationMsg(''); }}
            disabled={isLoading}
            autoComplete="off"
          />
        </div>
        <button type="submit" className="search-btn" disabled={isLoading} id="search-button">
          {isLoading ? (
            <span className="spinner" />
          ) : 'Search'}
        </button>
      </div>

      <button
        type="button"
        className="filter-toggle"
        onClick={() => setFiltersOpen(!filtersOpen)}
        id="filter-toggle"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
          <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
        </svg>
        Filters
        <span className={`chevron ${filtersOpen ? 'open' : ''}`}>›</span>
      </button>

      {filtersOpen && (
        <div className="filters-row">
          <div className="filter-group">
            <label htmlFor="sender-filter">Sender</label>
            <select id="sender-filter" value={sender} onChange={(e) => setSender(e.target.value)} disabled={isLoading}>
              <option value="">All senders</option>
              {SENDERS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="filter-group">
            <label htmlFor="date-start">From</label>
            <input id="date-start" type="date" value={dateStart} onChange={(e) => setDateStart(e.target.value)} disabled={isLoading} />
          </div>
          <div className="filter-group">
            <label htmlFor="date-end">To</label>
            <input id="date-end" type="date" value={dateEnd} onChange={(e) => setDateEnd(e.target.value)} disabled={isLoading} />
          </div>
        </div>
      )}

      {validationMsg && <p className="validation-msg">{validationMsg}</p>}
    </form>
  );
}
