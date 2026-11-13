"""Embeds a query, retrieves top-k chunks from Chroma, and converts distance to similarity."""


class Retriever:
    def __init__(self, embedder, store):
        # dependency injection: reuse existing Embedder/VectorStore instances so this
        # is easy to test with fakes instead of a real model + database
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.3,
        where: dict = None,
        debug: bool = False,
    ) -> list[dict]:

        query_embedding = self.embedder.embed([query])

        results = self.store.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
        )

        # Chroma's result lists are nested one level deep - one entry per query
        # embedding. We only ever send one, so index [0] everywhere.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        chunks = []

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Chroma returns cosine distance, not similarity
            score = 1 - distance

            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                **metadata,
                "score": score,
            })

        # print before filtering so borderline chunks that got dropped are
        # still visible - that's the case you actually want to debug
        if debug:
            print("\n========== RETRIEVAL DEBUG ==========")
            print(f"Query: {query!r}")
            print(f"k: {k} | Threshold: {threshold}")
            print(f"Retrieved chunks: {len(chunks)}")

            for i, chunk in enumerate(chunks, start=1):
                kept = "KEPT" if chunk["score"] >= threshold else "DROPPED"

                print(f"\nChunk {i}: [{kept}]")
                print(f"Score:  {chunk['score']:.4f}")
                print(f"Source: {chunk.get('source', 'Unknown')}")
                print(f"Page:   {chunk.get('page', 'N/A')}")
                print(f"Text:   {chunk['text'][:80]!r}")

            print("\n====================================\n")

        filtered_chunks = [
            chunk
            for chunk in chunks
            if chunk["score"] >= threshold
        ]

        # empty list (not None) is the signal callers rely on to skip the LLM call
        return filtered_chunks
