"""One-off analysis script (not shipped as part of src/) - measures retrieval hit
rate across chunk size, overlap, top-k, and chunking strategy. Results feed the
README's experiment table. Each config gets its own throwaway Chroma collection
under data/experiments/, so none of this touches the real data/chroma_db."""

import json
import uuid
from pathlib import Path

from src.chunker import chunk_document
from src.embedder import Embedder
from src.loader import load_pdfs, load_txt_or_md
from src.retriever import Retriever
from src.vector_store import VectorStore


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVAL_FILE = PROJECT_ROOT / "eval" / "test_questions.json"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiments"

# Keep retrieval threshold constant across experiments
THRESHOLD = 0.3


def load_test_questions() -> list[dict]:
    """Load evaluation questions from JSON."""

    with EVAL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_documents() -> list[dict]:
    """Load all supported documents from the raw directory."""

    records = load_txt_or_md(RAW_DIR)
    records.extend(load_pdfs(RAW_DIR))

    return records


def build_experiment_index(
    records: list[dict],
    embedder: Embedder,
    chunk_size: int,
    overlap: int,
    strategy: str,
):
    """
    Build a completely fresh Chroma collection for one experiment.

    Each experiment gets its own persistent directory so it
    cannot overwrite the production Chroma database.
    """

    # Create a unique experiment directory
    experiment_id = uuid.uuid4().hex[:8]

    experiment_path = (
        EXPERIMENT_DIR
        / f"{strategy}_{chunk_size}_{overlap}_{experiment_id}"
    )

    # Initialize a fresh vector store
    store = VectorStore(
        path=str(experiment_path),
        collection_name="experiment",
    )

    # Chunk ALL documents using the current configuration
    all_chunks = []

    for record in records:

        chunks = chunk_document(
            record,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(chunks)

    print(f"  Generated chunks: {len(all_chunks)}")

    # Generate embeddings and upsert into Chroma
    if all_chunks:

        texts = [chunk["text"] for chunk in all_chunks]

        embeddings = embedder.embed(texts)

        metadatas = []

        for chunk in all_chunks:

            metadata = {
                "doc_id": chunk["doc_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }

            if chunk["page"] is not None:
                metadata["page"] = chunk["page"]

            if chunk["char_start"] is not None:
                metadata["char_start"] = chunk["char_start"]

            metadatas.append(metadata)

        store.upsert(
            ids=[chunk["chunk_id"] for chunk in all_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"  Indexed chunks: {store.count()}")
    print(f"  Experiment database: {experiment_path}")

    return store


def run_experiment(
    records: list[dict],
    questions: list[dict],
    embedder: Embedder,
    chunk_size: int,
    overlap: int,
    strategy: str,
    k: int,
) -> dict:
    """
    Run one complete retrieval experiment.

    Measures whether at least one retrieved chunk comes
    from an expected source for each evaluation question.
    """

    print("\n" + "=" * 60)

    print(
        f"Experiment: size={chunk_size}, "
        f"overlap={overlap}, "
        f"strategy={strategy}, "
        f"k={k}"
    )

    print("=" * 60)

    # 1. Build a fresh index for this configuration
    store = build_experiment_index(
        records=records,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )

    # 2. Initialize retriever
    retriever = Retriever(embedder, store)

    # 3. Evaluate only questions with expected sources
    evaluable_questions = [
        q for q in questions
        if q["expected_sources"]
    ]

    hits = 0
    misses = 0

    question_results = []

    # 4. Retrieve and evaluate each question
    for question_data in evaluable_questions:

        question = question_data["question"]
        expected_sources = {
            Path(source.replace("\\", "/")).name.casefold()
            for source in question_data["expected_sources"]
        }

        retrieved_chunks = retriever.retrieve(
            query=question,
            k=k,
            threshold=THRESHOLD,
        )

        # compare by filename, not full path - metadata stores the full
        # "data/raw/..." path, not just the bare filename from the eval set
        retrieved_sources = {
            Path(chunk["source"].replace("\\", "/")).name.casefold()
            for chunk in retrieved_chunks
        }

        matched_sources = expected_sources & retrieved_sources

        hit = bool(matched_sources)

        if hit:
            hits += 1
        else:
            misses += 1

        question_results.append({
            "id": question_data["id"],
            "question": question,
            "expected_sources": sorted(expected_sources),
            "retrieved_sources": sorted(retrieved_sources),
            "matched_sources": sorted(matched_sources),
            "hit": hit,
        })

        status = "HIT" if hit else "MISS"

        print(
            f"  Q{question_data['id']:02d}: "
            f"{status} | "
            f"Expected: {sorted(expected_sources)}"
        )

    # 5. Calculate hit rate
    total_questions = len(evaluable_questions)

    hit_rate = (
        hits / total_questions
        if total_questions > 0
        else 0.0
    )

    results = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "strategy": strategy,
        "k": k,
        "threshold": THRESHOLD,
        "hits": hits,
        "misses": misses,
        "total_questions": total_questions,
        "hit_rate": hit_rate,
        "question_results": question_results,
    }

    print("\nExperiment results:")
    print(f"  Hits:     {hits}/{total_questions}")
    print(f"  Misses:   {misses}/{total_questions}")
    print(f"  Hit rate: {hit_rate:.2%}")

    return results


def main():

    # 1. Load documents and evaluation questions ONCE
    print("Loading documents and evaluation questions...")

    records = load_all_documents()
    questions = load_test_questions()

    print(f"Documents loaded: {len(records)}")
    print(f"Evaluation questions: {len(questions)}")

    # 2. Initialize embedding model ONCE
    print("\nLoading embedding model...")

    embedder = Embedder()

    # 3. Define experimental configurations
    # Change ONE variable at a time from the baseline.

    configs = [
        # Baseline
        (500, 50, "fixed", 4),

        # Chunk size experiments
        (200, 50, "fixed", 4),
        (1000, 50, "fixed", 4),

        # Overlap experiments
        (500, 0, "fixed", 4),
        (500, 100, "fixed", 4),

        # Top-k experiments
        (500, 50, "fixed", 2),
        (500, 50, "fixed", 8),

        # Chunking strategy experiment
        (500, 50, "recursive", 4),
    ]

    # 4. Run every experiment
    all_results = []

    for config in configs:

        result = run_experiment(
            records=records,
            questions=questions,
            embedder=embedder,
            chunk_size=config[0],
            overlap=config[1],
            strategy=config[2],
            k=config[3],
        )

        all_results.append(result)

    # 5. Print comparison summary
    print("\n\n")
    print("=" * 95)
    print("EXPERIMENT COMPARISON")
    print("=" * 95)

    print(
        f"{'Size':<8}"
        f"{'Overlap':<9}"
        f"{'Strategy':<12}"
        f"{'Top-k':<7}"
        f"{'Hits':<8}"
        f"{'Hit Rate':<10}"
        f"{'':<10}"
    )

    print("-" * 95)

    for result in all_results:
        bar_filled = round(result["hit_rate"] * 20)
        bar = "#" * bar_filled + "." * (20 - bar_filled)

        print(
            f"{result['chunk_size']:<8}"
            f"{result['overlap']:<9}"
            f"{result['strategy']:<12}"
            f"{result['k']:<7}"
            f"{str(result['hits']) + '/' + str(result['total_questions']):<8}"
            f"{result['hit_rate']:.0%}".ljust(10)
            + f"[{bar}]"
        )

    # 6. Save results to JSON
    results_file = EXPERIMENT_DIR / "experiment_results.json"

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nExperiments complete!")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
