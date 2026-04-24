import React from "react";

export default function ResponseView({ result, pureAnswer, loading }) {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">💬 Answer</div>
        <div className="card-body">
          <p className="answer-placeholder">Retrieving and generating…</p>
        </div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="card">
        <div className="card-header">💬 Answer</div>
        <div className="card-body">
          <p className="answer-placeholder">
            Your answer will appear here. Try asking about Ghana election results
            (1992–2020) or the 2025 budget.
          </p>
        </div>
      </div>
    );
  }

  const timings = result.stage_timings ?? {};
  const isMultistep = Boolean(result.multistep_trace);

  return (
    <div className="card">
      <div className="card-header">
        💬 Answer
        {isMultistep && <span className="multistep-badge">⚙ Multi-step</span>}
      </div>
      <div className="card-body">
        <p className="answer-text">{result.answer}</p>

        {/* Stage timings */}
        <div className="timings">
          {Object.entries(timings).map(([stage, ms]) => (
            <span key={stage} className="timing-chip">
              {stage}: {Math.round(ms)} ms
            </span>
          ))}
        </div>

        {/* Multi-step trace */}
        {isMultistep && result.multistep_trace?.computed_result && (
          <div style={{ marginTop: "0.7rem", fontSize: "0.83rem", color: "#374151" }}>
            <strong>Computed result:</strong> {result.multistep_trace.computed_result}
          </div>
        )}

        {/* Pure LLM comparison */}
        {pureAnswer && (
          <div className="comparison-block">
            <h4>Pure LLM answer (no retrieval):</h4>
            <p className="comparison-answer">{pureAnswer}</p>
          </div>
        )}
      </div>
    </div>
  );
}
