"""
OpenAI LLM wrapper.

Uses gpt-4o-mini via the OpenAI API.
Temperature 0.2 keeps answers consistent and reduces hallucination.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.2
_MAX_TOKENS = 512
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add your key to .env as OPENAI_API_KEY=sk-..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def generate(
    system_prompt: str,
    user_message: str,
    temperature: float = _TEMPERATURE,
    max_tokens: int = _MAX_TOKENS,
) -> str:
    """Call OpenAI chat completion and return the assistant message."""
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_no_retrieval(query: str) -> str:
    """
    Part E comparison: send query to OpenAI with no retrieved context.
    Used to measure hallucination rate of pure-LLM vs RAG.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the user's question based on your training knowledge."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()
