from embedder.store import get_collection, query_collection
from embedder.model import load_model

model = load_model()
col   = get_collection("vector_db", "docs")
print(f"Collection size: {col.count()} documents")
print("=" * 60)

questions = [
    "How do agents call tools?",
    "What authentication methods does GPT support?",
    "How does fine-tuning work?",
]

for q in questions:
    print(f"\nQ: {q}")
    print("-" * 60)
    results = query_collection(q, col, model, top_k=3)
    for i, r in enumerate(results, 1):
        dist = r["distance"]
        url  = r["source_url"]
        hdgs = r["headings"]
        txt  = r["text"]
        heading = " > ".join(hdgs.values()) if hdgs else "(no heading)"
        preview = txt[:120].strip()
        print(f"  [{i}] dist={dist:.4f}  {url}")
        print(f"       Section : {heading}")
        print(f"       Preview : {preview}...")
    print()
