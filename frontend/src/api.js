const BASE = import.meta.env.VITE_API_URL || "http://localhost:8001";

export async function queryRAG({ q, k = 5, mode = "hybrid", promptVersion = "v2" }) {
  const res = await fetch(`${BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q, k, mode, prompt_version: promptVersion }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export async function queryNoRetrieval(q) {
  const res = await fetch(`${BASE}/api/query/no-retrieval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}
