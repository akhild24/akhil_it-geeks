export default function QueryMeta({ queryType, filters }) {
  const typeLabels = {
    semantic: 'Semantic',
    attributed: 'Attributed',
    temporal: 'Temporal',
    mixed: 'Mixed',
  };

  const label = typeLabels[queryType] || queryType;

  const activeFilters = [];
  if (filters?.sender) activeFilters.push(`Sender: ${filters.sender}`);
  if (filters?.date_start) {
    const ds = new Date(filters.date_start).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    activeFilters.push(`From: ${ds}`);
  }
  if (filters?.date_end) {
    const de = new Date(filters.date_end).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    activeFilters.push(`To: ${de}`);
  }

  return (
    <div className="query-meta" id="query-meta">
      <span className="query-type-badge" data-type={queryType}>
        {label}
      </span>
      {activeFilters.map((f, i) => (
        <span key={i} className="filter-chip">{f}</span>
      ))}
    </div>
  );
}
