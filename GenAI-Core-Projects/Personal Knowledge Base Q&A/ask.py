"""Interactive CLI Q&A loop - wires Retriever + generator together, extending the
Week 9 chatbot loop (system prompt -> history -> token awareness -> logging) with
a retrieval step."""

import argparse
import time

from src.embedder import Embedder
from src.generator import build_context, generate_answer
from src.logger import ConversationLogger
from src.retriever import Retriever
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your personal knowledge base."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieval debugging information."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)."
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    # construct these once, not per-question - loading the embedding model
    # on every turn would waste real time
    print("Loading knowledge base...")

    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store)
    logger = ConversationLogger(model="gpt-4o-mini")

    print(f"Knowledge base loaded: {store.count()} chunks.")
    print("Ask a question about your notes.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # time retrieval and generation separately so the logger can report
        # where the latency actually goes
        start_time = time.perf_counter()

        chunks = retriever.retrieve(
            query=question,
            k=args.k,
            threshold=args.threshold,
            debug=args.debug,
        )

        retrieval_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # nothing cleared the similarity threshold - skip the LLM call entirely
        if not chunks:
            answer = "I couldn't find this in your notes."
            llm_ms = 0.0

            print(f"\nAssistant: {answer}\n")

            logger.log_qa_turn(
                question=question,
                answer=answer,
                chunks=chunks,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
            )

            continue

        context = build_context(
            chunks,
            max_tokens=3000,
        )

        # chunks were retrieved but none fit the token budget - a different
        # failure mode than "not found", so it gets its own message
        if not context:
            answer = (
                "Relevant notes were retrieved, but they exceeded "
                "the context token budget. No answer was generated."
            )

            llm_ms = 0.0

            print(f"\nAssistant: {answer}\n")

            logger.log_qa_turn(
                question=question,
                answer=answer,
                chunks=chunks,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
            )

            continue

        start_time = time.perf_counter()

        answer = generate_answer(
            question=question,
            context=context,
        )

        llm_ms = (
            time.perf_counter() - start_time
        ) * 1000

        print(f"\nAssistant: {answer}\n")

        logger.log_qa_turn(
            question=question,
            answer=answer,
            chunks=chunks,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
        )

    # Save session when the loop exits
    logger.save()


if __name__ == "__main__":
    main()
