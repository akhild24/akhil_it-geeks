const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function searchMessages({ query, sender, dateStart, dateEnd, topK = 10 }) {
  const body = {
    query,
    top_k: topK,
  };

  if (sender && sender !== '') {
    body.sender = sender;
  }
  if (dateStart) {
    body.date_start = new Date(dateStart).toISOString();
  }
  if (dateEnd) {
    // Set to end of day
    const d = new Date(dateEnd);
    d.setHours(23, 59, 59, 999);
    body.date_end = d.toISOString();
  }

  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => null);
    const detail = errorData?.detail || `Request failed with status ${res.status}`;
    throw new Error(detail);
  }

  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend unavailable');
  return res.json();
}
