export default function SourceList({ sources }) {
  return (
    <details className="sources">
      <summary>Sources</summary>
      {sources.map((s, i) => (
        <div key={i} className="source-item">
          <strong>{s.id}</strong> (score: {s.score})
          <div>{s.text}</div>
        </div>
      ))}
    </details>
  );
}
