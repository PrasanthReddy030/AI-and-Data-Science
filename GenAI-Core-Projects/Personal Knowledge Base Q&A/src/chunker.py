"""Splits document text into chunks (two strategies) and attaches retrieval metadata.

chunk_fixed is hand-rolled: a sliding token window over tiktoken's cl100k_base encoding.
chunk_recursive uses LangChain's RecursiveCharacterTextSplitter - the one piece of this
project that isn't built from scratch (everything else - loader, fixed chunking, embedder,
vector store, retriever, generator - has no LangChain dependency).
"""

import hashlib

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Token-based sliding window with overlap, decoded back to text per chunk."""
    assert chunk_size > 0, "chunk_size must be positive"
    assert 0 <= overlap < chunk_size, "overlap must be >= 0 and smaller than chunk_size"

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk_tokens))

    return chunks


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50, separators: list[str] = None) -> list[str]:
    """Paragraph-aware splitting (falls back to smaller separators, then raw tokens)
    via LangChain's RecursiveCharacterTextSplitter, measured in cl100k_base tokens."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", ""]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
    )

    return text_splitter.split_text(text)


def chunk_document(
    record: dict,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """Chunk one loader record and attach chunk_id/doc_id/source/page/chunk_index/char_start."""
    text = record["text"]

    if strategy == "fixed":
        raw_chunks = chunk_fixed(text, chunk_size, overlap)

    elif strategy == "recursive":
        raw_chunks = chunk_recursive(text, chunk_size, overlap)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chunks = []
    search_from = 0

    for i, chunk_text in enumerate(raw_chunks):
        # advance the search cursor past this match so an identical chunk
        # earlier in the text doesn't get matched again
        char_start = text.find(chunk_text, search_from)

        if char_start != -1:
            search_from = char_start + len(chunk_text)

        # page is part of the hash because a PDF's pages share the same source path -
        # without it, identical boilerplate text on two pages would collide on chunk_id
        unique_string = (
            f"{record['source']}|"
            f"{record.get('page')}|"
            f"{chunk_text}"
        )

        chunk_id = hashlib.sha256(
            unique_string.encode("utf-8")
        ).hexdigest()

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": record["doc_id"],
            "source": record["source"],
            "page": record.get("page"),
            "chunk_index": i,
            "char_start": char_start if char_start != -1 else None,
            "text": chunk_text,
        })

    return chunks