
from unittest.mock import patch, MagicMock

from src.retriever import Retriever
from src.generator import build_context, generate_answer


class FakeEmbedder:
    """Avoid loading the real SentenceTransformer model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    """Return predictable Chroma-style retrieval results."""

    def query(
        self,
        query_embeddings,
        n_results=5,
        where=None,
    ):

        return {
            "ids": [[
                "test_chunk_1",
                "test_chunk_2",
            ]],

            "documents": [[
                "Chroma is an open-source embedding database.",
                "Chroma supports vector search and retrieval.",
            ]],

            "metadatas": [[
                {
                    "doc_id": "chroma_README",
                    "source": "chroma_README.md",
                    "chunk_index": 0,
                },
                {
                    "doc_id": "chroma_README",
                    "source": "chroma_README.md",
                    "chunk_index": 1,
                },
            ]],

            "distances": [[
                0.1,
                0.2,
            ]],
        }


def test_ask_flow_with_mocked_llm():
    """
    Test the retrieval-to-generation pipeline.

    Real retrieval, context construction and generation
    logic are exercised, but the OpenAI API is mocked.
    """

    # 1. Initialize fake retrieval dependencies
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    retriever = Retriever(embedder, store)

    question = "What is Chroma?"

    # 2. Retrieve relevant chunks using the real Retriever
    chunks = retriever.retrieve(
        query=question,
        k=2,
        threshold=0.3,
    )

    # Both chunks should pass the threshold
    assert len(chunks) == 2

    # 3. Build context using the real context builder
    context = build_context(
        chunks,
        max_tokens=3000,
    )

    assert context
    assert "Chroma is an open-source embedding database." in context
    assert "chroma_README.md" in context

    # 4. Create a fake OpenAI response
    fake_response = MagicMock()

    fake_response.choices[0].message.content = (
        "Chroma is an open-source embedding database "
        "that supports vector search and retrieval. "
        "[chroma_README.md, p.N/A]"
    )

    # 5. Mock the actual OpenAI API call
    with patch(
        "src.generator.client.chat.completions.create",
        return_value=fake_response,
    ) as mock_create:

        answer = generate_answer(
            question=question,
            context=context,
        )

    # 6. Verify the generated answer
    assert isinstance(answer, str)

    assert "Chroma" in answer
    assert answer == fake_response.choices[0].message.content

    # 7. Verify the OpenAI API mock was actually called
    mock_create.assert_called_once()

    # 8. Inspect the messages sent to the mocked API
    call_kwargs = mock_create.call_args.kwargs

    messages = call_kwargs["messages"]

    # Check that the context reached the LLM
    all_message_content = " ".join(
        message["content"]
        for message in messages
    )

    assert "Chroma is an open-source embedding database." in (
        all_message_content
    )

    assert question in all_message_content
