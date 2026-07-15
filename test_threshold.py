from embedder.store import get_collection, query_collection
from embedder.model import load_model

model = load_model()
col   = get_collection("vector_db", "docs")

questions = {
    # --- Clearly IN scope (OpenAI API docs) ---
    "IN":  [
        "How do agents call tools?",
        "What authentication methods does GPT support?",
        "How does fine-tuning work?",
    ],
    # --- Clearly OUT of scope ---
    "OUT": [
        "What is the boiling point of mercury?",
        "How do I file taxes in India?",
        "Who won the FIFA World Cup in 2022?",
        "What is the speed of light?",
        "How do I cook biryani?",
    ],
    # --- Borderline / adjacent (tech but not in docs) ---
    "EDGE": [
        "How does Kubernetes handle pod scheduling?",
        "What is a transformer neural network?",
        "How does Python's GIL work?",
    ],
}

for label, qs in questions.items():
    print(f"\n{'='*60}")
    print(f"  {label} SCOPE")
    print(f"{'='*60}")
    for q in qs:
        results = query_collection(q, col, model, top_k=3)
        dists = [r["distance"] for r in results]
        best  = results[0]
        print(f"\n  Q: {q}")
        print(f"     distances : {[f'{d:.4f}' for d in dists]}")
        print(f"     best match: {best['source_url']}")
        print(f"     section   : {' > '.join(best['headings'].values()) if best['headings'] else '(none)'}")
