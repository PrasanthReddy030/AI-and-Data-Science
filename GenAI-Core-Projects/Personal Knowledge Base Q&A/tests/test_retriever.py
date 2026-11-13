
from src.retriever import Retriever


class FakeEmbedder:
    """Fake embedder that avoids loading a real embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    """Fake vector store returning predictable Chroma-style results."""

    def __init__(self):
        self.results = {
            "ids": [[
                "chunk_1",
                "chunk_2",
                "chunk_3",
            ]],
            "documents": [[
                "This is highly relevant content.",
                "This is moderately relevant content.",
                "This is less relevant content.",
            ]],
            "metadatas": [[
                {
                    "source": "document1.md",
                    "doc_id": "document1",
                    "chunk_index": 0,
                },
                {
                    "source": "document2.pdf",
                    "doc_id": "document2",
                    "chunk_index": 1,
                    "page": 2,
                },
                {
                    "source": "document3.txt",
                    "doc_id": "document3",
                    "chunk_index": 2,
                },
            ]],
            # Chroma cosine distances:
            # Similarity = 1 - distance
            "distances": [[
                0.1,  # Similarity: 0.9
                0.3,  # Similarity: 0.7
                0.6,  # Similarity: 0.4
            ]],
        }

        self.last_query_embeddings = None
        self.last_n_results = None
        self.last_where = None

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict = None,
    ) -> dict:

        self.last_query_embeddings = query_embeddings
        self.last_n_results = n_results
        self.last_where = where

        return self.results


def test_retrieve_filters_below_threshold():
    """Chunks below the similarity threshold should be excluded."""

    embedder = FakeEmbedder()
    store = FakeVectorStore()

    retriever = Retriever(embedder, store)

    chunks = retriever.retrieve(
        query="What is relevant?",
        k=3,
        threshold=0.7,
    )

    # Expected similarities:
    # chunk_1 = 0.9 -> KEPT
    # chunk_2 = 0.7 -> KEPT (threshold is inclusive)
    # chunk_3 = 0.4 -> DROPPED

    assert len(chunks) == 2

    returned_ids = [
        chunk["chunk_id"] for chunk in chunks
    ]

    assert returned_ids == ["chunk_1", "chunk_2"]


def test_retrieve_returns_empty_list_when_nothing_clears_threshold():
    """No qualifying chunks should produce an empty list."""

    retriever = Retriever(
        FakeEmbedder(),
        FakeVectorStore(),
    )

    chunks = retriever.retrieve(
        query="Something unrelated",
        k=3,
        threshold=0.95,
    )

    assert chunks == []
    assert isinstance(chunks, list)


def test_retrieve_converts_distance_to_similarity_correctly():
    """Chroma distance should be converted into similarity."""

    retriever = Retriever(
        FakeEmbedder(),
        FakeVectorStore(),
    )

    chunks = retriever.retrieve(
        query="Test query",
        k=3,
        threshold=0.0,
    )

    # First chunk has distance 0.1
    # Similarity = 1 - 0.1 = 0.9

    assert chunks[0]["score"] == 0.9

    # Second chunk has distance 0.3
    assert chunks[1]["score"] == 0.7

    # Third chunk has distance 0.6
    assert chunks[2]["score"] == 0.4


def test_retrieve_passes_query_parameters_to_store():
    """Retriever should pass k, embeddings and metadata filters."""

    embedder = FakeEmbedder()
    store = FakeVectorStore()

    retriever = Retriever(embedder, store)

    retriever.retrieve(
        query="Find relevant information",
        k=2,
        threshold=0.0,
        where={"doc_id": "document1"},
    )

    assert store.last_n_results == 2

    assert store.last_query_embeddings == [
        [0.1, 0.2, 0.3]
    ]

    assert store.last_where == {
        "doc_id": "document1"
    }
