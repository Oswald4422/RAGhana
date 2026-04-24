import React, { useState } from "react";

export default function PromptView({ finalPrompt }) {
  const [open, setOpen] = useState(false);
  if (!finalPrompt) return null;

  return (
    <div className="card">
      <div className="card-header">🧪 Prompt Inspector</div>
      <div className="card-body">
        <button className="prompt-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? "Hide final prompt ▲" : "Show final prompt ▼"}
        </button>
        {open && (
          <pre className="prompt-box">{finalPrompt}</pre>
        )}
      </div>
    </div>
  );
}
