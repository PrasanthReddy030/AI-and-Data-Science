"""CLI: builds the Chroma collection from data/raw. Idempotent - hashes each file's
contents so unchanged files are skipped, and re-ingests only new or modified files,
deleting their stale chunks first."""

import hashlib
import json
from pathlib import Path

from src.chunker import chunk_document
from src.embedder import Embedder
from src.loader import load_pdfs, load_txt_or_md
from src.vector_store import VectorStore

HASH_FILE = Path("data/ingest_hashes.json")


def load_hash_cache() -> dict:
    if HASH_FILE.exists():
        with HASH_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_hash_cache(cache: dict):
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)

    with HASH_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)


def file_hash(filepath: Path) -> str:
    sha256 = hashlib.sha256()

    with filepath.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def main():
    raw_dir = Path("data/raw")
    old_cache = load_hash_cache()

    # 2. Calculate hashes for current files (once per file)
    current_hashes = {}

    for file in sorted(raw_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in {
            ".txt", ".md", ".pdf"
        }:
            current_hashes[str(file)] = file_hash(file)

    # 3. Identify changed, new and deleted sources
    #
    # four possible file states, and what each one means for ingestion:
    #   new file                          -> ingest it
    #   existing file, unchanged hash      -> skip it
    #   existing file, changed hash        -> delete old chunks, re-ingest
    #   previously ingested, now deleted   -> delete old chunks, drop from cache
    changed_sources = {
        source
        for source, new_hash in current_hashes.items()
        if old_cache.get(source) != new_hash
    }

    deleted_sources = set(old_cache) - set(current_hashes)

    print(f"New/changed files: {len(changed_sources)}")
    print(f"Deleted files: {len(deleted_sources)}")
    print(
        f"Unchanged files: "
        f"{len(current_hashes) - len(changed_sources)}"
    )

    # 4. Initialize vector store
    store = VectorStore()

    # Delete old chunks for changed or deleted sources
    sources_to_delete = changed_sources | deleted_sources

    for source in sources_to_delete:
        store.delete_by_source(source)
        print(f"Deleted old chunks: {source}")

    # 5. Load documents
    records = load_txt_or_md(raw_dir)
    records.extend(load_pdfs(raw_dir))

    # Only process records belonging to changed/new files
    records = [
        record
        for record in records
        if record["source"] in changed_sources
    ]

    print(f"Records to process: {len(records)}")

    # 6. Chunk documents
    all_chunks = []

    for record in records:
        all_chunks.extend(
            chunk_document(
                record,
                strategy="fixed",
                chunk_size=500,
                overlap=50,
            )
        )

    print(f"Generated chunks: {len(all_chunks)}")

    # 7. Embed and upsert
    if all_chunks:

        embedder = Embedder()

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

    # 8. Persist updated hashes
    # Only reached if ingestion has completed successfully.
    save_hash_cache(current_hashes)

    print("Ingestion complete!")
    print(f"Total chunks in ChromaDB: {store.count()}")


if __name__ == "__main__":
    main()
