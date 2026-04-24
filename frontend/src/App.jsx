import React, { useState } from "react";
import QueryInput from "./components/QueryInput.jsx";
import RetrievedChunks from "./components/RetrievedChunks.jsx";
import ResponseView from "./components/ResponseView.jsx";
import PromptView from "./components/PromptView.jsx";
import { queryRAG, queryNoRetrieval } from "./api.js";

export default function App() {
  const [result, setResult] = useState(null);
  const [pureAnswer, setPureAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (params) => {
    setLoading(true);
    setError(null);
    setPureAnswer(null);
    try {
      const data = await queryRAG(params);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async (q) => {
    setLoading(true);
    setError(null);
    setPureAnswer(null);
    try {
      // Run RAG first if we don't have a result for this query
      const ragData = await queryRAG({ q });
      setResult(ragData);
      // Then run pure LLM
      const pure = await queryNoRetrieval(q);
      setPureAnswer(pure.answer);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <span className="header-flag">🇬🇭</span>
        <div>
          <h1>RAGhana</h1>
          <p>AI assistant for Ghana election results (1992–2020) &amp; 2025 Budget Statement</p>
        </div>
      </header>

      {/* Main layout */}
      <main className="main">
        {/* Left column: query + response + prompt */}
        <div className="left-col">
          <QueryInput onSubmit={handleSubmit} onCompare={handleCompare} loading={loading} />

          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
            </div>
          )}

          <ResponseView result={result} pureAnswer={pureAnswer} loading={loading} />
          <PromptView finalPrompt={result?.final_prompt} />
        </div>

        {/* Right column: retrieved chunks */}
        <div className="right-col">
          <RetrievedChunks chunks={result?.retrieved_chunks} />
        </div>
      </main>

      <footer className="footer">
        RAGhana · Academic City University · No LangChain · No LlamaIndex · Built from scratch
      </footer>
    </div>
  );
}
