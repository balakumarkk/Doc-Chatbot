"""
chat/llm.py
-----------
Groq API client + prompt construction for RAG-grounded answers.

Uses the OpenAI-compatible client pointed at Groq's base URL.
GROQ_API_KEY is loaded from the .env file via python-dotenv.

Key design decisions:
  - Temperature 0.2  : factual RAG answers, not creative writing
  - Numbered context : LLM cites [1], [2] inline for traceability
  - "I don't know"   : explicit instruction prevents hallucination
  - Streaming        : tokens printed as they arrive for responsiveness
"""

from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv

# Load .env at module import time
load_dotenv()

DEFAULT_MODEL       = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 1024
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"

_SYSTEM_PROMPT = """You are a documentation assistant. Your job is to answer questions \
accurately using ONLY the context provided below.

Rules:
- Answer based solely on the provided context. Do not use outside knowledge.
- Cite sources inline using [1], [2], etc. corresponding to the context blocks.
- If the context does not contain enough information to answer the question, respond with:
  "I don't have enough information in the documentation to answer that."
- Be concise and direct. Do not pad your answer.
- Do not make up URLs, function names, or parameters not present in the context."""


def _get_client():
    """Build an OpenAI-compatible client pointed at Groq."""
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )

    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for the Groq API call.

    Parameters
    ----------
    question : User's question.
    chunks   : Retrieved chunks from retrieve() — may be empty.

    Returns
    -------
    List of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
    """
    if not chunks:
        # No relevant context — let the system prompt handle "I don't know"
        context_text = "(No relevant documentation found for this question.)"
    else:
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            heading = " > ".join(chunk["headings"].values()) if chunk.get("headings") else ""
            block = f"[{i}] Source: {chunk['source_url']}"
            if heading:
                block += f"\nSection: {heading}"
            block += f"\n---\n{chunk['text']}"
            blocks.append(block)
        context_text = "\n\n".join(blocks)

    user_message = f"Context:\n\n{context_text}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def stream_answer(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[str]:
    """
    Call Groq API with streaming. Yields text tokens as they arrive.

    Parameters
    ----------
    messages    : Output of build_prompt().
    model       : Groq model ID.
    temperature : Sampling temperature (0.2 = factual/consistent).
    max_tokens  : Max output tokens.

    Yields
    ------
    str — text delta tokens as they stream in.
    """
    client = _get_client()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_answer(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Non-streaming version — returns the full answer as a string.
    Useful for programmatic use or testing.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    return response.choices[0].message.content
