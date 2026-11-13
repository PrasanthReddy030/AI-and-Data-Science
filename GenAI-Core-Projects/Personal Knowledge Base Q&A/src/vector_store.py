"""Thin wrapper around a persistent Chroma collection. Embeddings are always supplied
by the caller - no default embedding function is configured on the collection."""

import chromadb


class VectorStore:
    def __init__(
        self,
        path: str = "data/chroma_db",
        collection_name: str = "personal_kb"
    ):
        self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict = None,
    ) -> dict:

        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )
    
    def count(self) -> int:
        return self.collection.count()

    def delete_by_source(self, source: str):
        """Delete every chunk whose metadata source matches this path - used by
        ingest.py to clear stale chunks before re-ingesting a changed file."""
        self.collection.delete(where={"source": source})
