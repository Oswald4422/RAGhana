import React, { useState } from "react";

export default function QueryInput({ onSubmit, onCompare, loading }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [promptVersion, setPromptVersion] = useState("v2");
  const [k, setK] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!q.trim() || loading) return;
    onSubmit({ q: q.trim(), mode, promptVersion, k });
  };

  const handleCompare = () => {
    if (!q.trim() || loading) return;
    onCompare(q.trim());
  };

  return (
    <div className="card">
      <div className="card-header">🔍 Ask RAGhana</div>
      <div className="card-body">
        <form className="query-form" onSubmit={handleSubmit}>
          <textarea
            className="query-textarea"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. How many votes did NPP receive in Ashanti Region in 2020?&#10;e.g. What is Ghana's debt-to-GDP target for 2025?"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <div className="query-controls">
            <label>
              Mode&nbsp;
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="hybrid">Hybrid (BM25 + Vector)</option>
                <option value="vector">Vector only</option>
                <option value="bm25">BM25 only</option>
              </select>
            </label>
            <label>
              Prompt&nbsp;
              <select value={promptVersion} onChange={(e) => setPromptVersion(e.target.value)}>
                <option value="v2">v2 – Grounded + cite</option>
                <option value="v1">v1 – Naïve baseline</option>
                <option value="v3">v3 – + Few-shot</option>
              </select>
            </label>
            <label>
              k&nbsp;
              <select value={k} onChange={(e) => setK(Number(e.target.value))}>
                {[3, 5, 8, 10].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="compare-btn"
              onClick={handleCompare}
              disabled={!q.trim() || loading}
              title="Run same query without retrieval (Part E comparison)"
            >
              vs Pure LLM
            </button>
            <button type="submit" className="send-btn" disabled={!q.trim() || loading}>
              {loading ? <><span className="spinner" />Thinking…</> : "Send →"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
