
import tiktoken

from src.chunker import chunk_fixed, chunk_recursive


# Use the same tokenizer as chunker.py
enc = tiktoken.get_encoding("cl100k_base")


def test_chunk_fixed_respects_overlap():
    """Consecutive fixed chunks should share the requested tokens."""

    text = " ".join(f"word{i}" for i in range(1000))

    chunk_size = 100
    overlap = 20

    chunks = chunk_fixed(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    token_chunks = [
        enc.encode(chunk) for chunk in chunks
    ]

    assert len(token_chunks) > 1

    for i in range(len(token_chunks) - 1):
        previous_tokens = token_chunks[i]
        next_tokens = token_chunks[i + 1]

        assert (
            previous_tokens[-overlap:]
            == next_tokens[:overlap]
        )


def test_chunk_fixed_no_empty_chunks():
    """Fixed chunking should never produce empty chunks."""

    chunk_size = 100
    overlap = 20
    step = chunk_size - overlap

    # Create text with exactly four steps worth of tokens
    tokens = enc.encode(
        " ".join(f"word{i}" for i in range(1000))
    )

    exact_tokens = tokens[:step * 4]
    text = enc.decode(exact_tokens)

    assert len(enc.encode(text)) == step * 4

    chunks = chunk_fixed(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_fixed_zero_overlap_still_works():
    """Zero overlap should produce valid non-overlapping chunks."""

    text = " ".join(f"word{i}" for i in range(500))

    chunk_size = 100

    chunks = chunk_fixed(
        text,
        chunk_size=chunk_size,
        overlap=0,
    )

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)

    # Every chunk should contain at most chunk_size tokens
    for chunk in chunks:
        assert len(enc.encode(chunk)) <= chunk_size


def test_chunk_recursive_no_empty_chunks():
    """Recursive chunking should skip empty and whitespace-only chunks."""

    text = (
        "\n\n\n"
        + "\n\n".join(
            f"Paragraph {i}. Some example content here."
            for i in range(100)
        )
        + "\n\n\n"
    )

    chunks = chunk_recursive(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_recursive_respects_chunk_size_roughly():
    """
    Recursive chunks should generally stay within chunk_size.

    A small allowance is included for tokenizer boundary
    differences after decoding and re-encoding.
    """

    text = "\n\n".join(
        f"Paragraph {i}. "
        "This paragraph contains some example text "
        "to test recursive splitting behavior."
        for i in range(100)
    )

    chunk_size = 100

    chunks = chunk_recursive(
        text,
        chunk_size=chunk_size,
        overlap=20,
    )

    assert len(chunks) > 0

    for chunk in chunks:
        token_count = len(enc.encode(chunk))

        assert token_count <= chunk_size + 5, (
            f"Chunk exceeded expected size: "
            f"{token_count} tokens"
        )
