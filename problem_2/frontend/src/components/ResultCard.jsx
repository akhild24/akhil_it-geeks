import { useState } from 'react';

// Deterministic sender color palette
const SENDER_COLORS = {
  priya: '#e07cda',
  meera: '#6ec6ff',
  aditya: '#ffb347',
  neha: '#77dd77',
  simran: '#ff6b6b',
  kunal: '#c3a6ff',
  akhil: '#54d6bb',
  rohan: '#ffd700',
};

function getSenderColor(sender) {
  return SENDER_COLORS[sender?.toLowerCase()] || 'var(--text-muted)';
}

function formatTime(ts) {
  try {
    const d = new Date(ts);
    return d.toLocaleString('en-IN', {
      day: 'numeric', month: 'short',
      hour: '2-digit', minute: '2-digit', hour12: true,
    });
  } catch {
    return ts;
  }
}

export default function ResultCard({ result, rank }) {
  const [contextOpen, setContextOpen] = useState(true);

  return (
    <div className="result-card" id={`result-${result.message_id}`}>
      <div className="result-header">
        <span className="result-rank">#{rank}</span>
        <span className="result-sender" style={{ color: getSenderColor(result.sender) }}>
          {result.sender}
        </span>
        <span className="result-time">{formatTime(result.timestamp)}</span>
        <span className="result-score" title={`BM25 rank: ${result.bm25_rank ?? '–'}  ·  Dense rank: ${result.dense_rank ?? '–'}`}>
          score {result.fused_score.toFixed(4)}
        </span>
      </div>

      <p className="result-text">{result.text}</p>

      {result.context && result.context.length > 0 && (
        <>
          <button
            type="button"
            className="context-toggle"
            onClick={() => setContextOpen(!contextOpen)}
          >
            {contextOpen ? '▾ Hide context' : '▸ Show context'}
          </button>

          {contextOpen && (
            <div className="context-thread">
              {result.context.map((msg) => (
                <div
                  key={msg.id}
                  className={`context-msg ${msg.is_match ? 'context-match' : ''}`}
                >
                  <div className="context-msg-header">
                    <span className="context-sender" style={{ color: getSenderColor(msg.sender) }}>
                      {msg.sender}
                    </span>
                    <span className="context-time">{formatTime(msg.timestamp)}</span>
                    {msg.is_match && <span className="match-label">MATCH</span>}
                  </div>
                  <p className="context-text">{msg.text}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
