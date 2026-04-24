import React, { useState } from "react";

function ChunkItem({ chunk, index }) {
  const [expanded, setExpanded] = useState(false);
  const score = chunk.score ?? 0;
  const isHighScore = score > 0.7;
  const sourceClass = chunk.source === "csv" ? "chunk-source csv" : "chunk-source";
  const sourceLabel = chunk.source === "csv"
    ? `CSV ${chunk.metadata?.year ?? ""}/${chunk.metadata?.region ?? ""}`
    : `PDF p${chunk.metadata?.page_approx ?? chunk.metadata?.page_start ?? "?"}`;

  return (
    <div className="chunk-item">
      <div className="chunk-header" onClick={() => setExpanded((v) => !v)}>
        <span className="chunk-id">C{index + 1}</span>
        <span className={sourceClass}>{sourceLabel}</span>
        <span className={`score-chip${isHighScore ? " high" : ""}`}>
          {score.toFixed(3)}
        </span>
        <span className="expand-icon">{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="chunk-body">{chunk.text}</div>
      )}
    </div>
  );
}

export default function RetrievedChunks({ chunks }) {
  if (!chunks) {
    return (
      <div className="card">
        <div className="card-header">📄 Retrieved Chunks</div>
        <div className="card-body">
          <p className="no-chunks">No results yet — send a query first.</p>
        </div>
      </div>
    );
  }
  if (chunks.length === 0) {
    return (
      <div className="card">
        <div className="card-header">📄 Retrieved Chunks</div>
        <div className="card-body">
          <p className="no-chunks">No chunks retrieved.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="card">
      <div className="card-header">📄 Retrieved Chunks ({chunks.length})</div>
      <div className="card-body">
        <div className="chunk-list">
          {chunks.map((chunk, i) => (
            <ChunkItem key={chunk.id ?? i} chunk={chunk} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
