#!/usr/bin/env python3
"""
chat.py — Doc Chatbot CLI
=========================
Ask questions about your scraped documentation.
Retrieves relevant chunks from the vector DB, then uses Groq (Llama 3.1 70B)
to generate a grounded answer with source citations.

Usage
-----
  # Interactive mode (default)
  python chat.py

  # Single question
  python chat.py --question "How do agents call tools?"

  # Adjust retrieval
  python chat.py --top-k 3 --threshold 0.35

  # Non-streaming (useful for piping output)
  python chat.py --question "..." --no-stream
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat",
        description="RAG chatbot over your scraped documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--question", "-q",
        metavar="TEXT",
        default=None,
        help="Ask a single question and exit (default: interactive loop).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="INT",
        help="Number of chunks to retrieve (default: 5).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        metavar="FLOAT",
        help="Distance threshold — chunks beyond this are dropped (default: 0.38).",
    )
    p.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        metavar="NAME",
        help="Groq model ID (default: llama-3.1-70b-versatile).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="Sampling temperature (default: 0.2).",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Print full answer at once instead of streaming.",
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        default=True,
        help="Print source URLs after the answer (default: true).",
    )
    p.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        help="Hide source URLs.",
    )
    p.add_argument(
        "--db-path",
        default="vector_db",
        metavar="DIR",
        help="Chroma DB directory (default: vector_db/).",
    )
    p.add_argument(
        "--collection",
        default="docs",
        metavar="NAME",
        help="Chroma collection name (default: docs).",
    )
    return p


# ---------------------------------------------------------------------------
# Answer one question
# ---------------------------------------------------------------------------
def answer_question(question: str, args: argparse.Namespace) -> None:
    from chat.retriever import retrieve
    from chat.llm import build_prompt, stream_answer, get_answer

    # --- Retrieve ---
    chunks = retrieve(
        question,
        top_k=args.top_k,
        distance_threshold=args.threshold,
        db_path=args.db_path,
        collection_name=args.collection,
    )

    # --- Build prompt ---
    messages = build_prompt(question, chunks)

    # --- Generate answer ---
    print()  # blank line before answer

    if args.no_stream:
        answer = get_answer(messages, model=args.model, temperature=args.temperature)
        print(answer)
    else:
        for token in stream_answer(messages, model=args.model, temperature=args.temperature):
            print(token, end="", flush=True)
        print()  # newline after streamed answer

    # --- Source citations ---
    if args.show_sources and chunks:
        seen_urls = []
        for chunk in chunks:
            url = chunk.get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.append(url)

        if seen_urls:
            print("\nSources:")
            for i, url in enumerate(seen_urls, 1):
                print(f"  [{i}] {url}")

    print()  # trailing blank line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    if args.question:
        # Single question mode
        answer_question(args.question, args)
    else:
        # Interactive loop
        print("Doc Chatbot — powered by Groq + Llama 3.1 70B")
        print(f"Corpus: {args.db_path}/{args.collection}  |  threshold={args.threshold}  |  top_k={args.top_k}")
        print("Type your question and press Enter. Type 'quit' or Ctrl+C to exit.")
        print("-" * 60)

        while True:
            try:
                question = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q", "bye"):
                print("Bye!")
                break

            answer_question(question, args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run(args)
