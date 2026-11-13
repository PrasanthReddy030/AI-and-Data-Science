"""One-off script that runs the real production pipeline against eval/test_questions.json
and prints each answer for manual grading. Does not grade anything itself - correct/
partial/hallucinated is a judgment call recorded separately in eval/grading_results.json
after reading this output."""

import json
from pathlib import Path

from src.embedder import Embedder
from src.generator import build_context, generate_answer
from src.retriever import Retriever
from src.vector_store import VectorStore

EVAL_FILE = Path("eval/test_questions.json")

# same defaults ask.py uses - this should reflect whatever you'd actually run in production,
# against your real data/chroma_db (already built by ingest.py), not a throwaway experiment collection
K = 4
THRESHOLD = 0.3


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    embedder = Embedder()
    store = VectorStore()  # real production collection, not an experiment one
    retriever = Retriever(embedder, store)

    for q in questions:
        print("=" * 80)
        print(f"Q{q['id']}: {q['question']}")
        print(f"expected_sources: {q['expected_sources']}")
        print(f"notes: {q.get('notes', '')}")
        print("-" * 80)

        chunks = retriever.retrieve(query=q["question"], k=K, threshold=THRESHOLD)

        if not chunks:
            print("RETRIEVAL: no chunks cleared the threshold - not found path, no LLM call")
            print()
            continue

        retrieved_sources = sorted({Path(c["source"]).name for c in chunks})
        print(f"RETRIEVED SOURCES: {retrieved_sources}")

        context = build_context(chunks, max_tokens=3000)
        answer = generate_answer(question=q["question"], context=context)

        print(f"ANSWER:\n{answer}")
        print()


if __name__ == "__main__":
    main()
