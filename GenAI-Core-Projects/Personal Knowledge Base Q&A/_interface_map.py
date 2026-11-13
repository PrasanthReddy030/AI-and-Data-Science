# SCRATCH FILE - not part of the project, just a running map of function signatures
# so wiring things up later doesn't need re-reading every module. Delete before final push.
# Real stub syntax below (not meant to run) so the editor actually syntax-highlights it.


# ── src/loader.py ──────────────────────────────────────────────────────────

def load_txt_or_md(dirpath: "Path") -> "list[dict]":
    """One dict per .txt/.md file found directly in dirpath."""
    ...

# example record returned (one entry in the list):
load_txt_or_md__example = {
    "doc_id": "chroma_README",
    "source": "data/raw/chroma_README.md",
    "text": "# Chroma\n\nthe AI-native open-source embedding database...",
    "page": None,
}


def load_pdfs(dirpath: "Path") -> "list[dict]":
    """One dict per non-empty PAGE across all .pdf files in dirpath."""
    ...

# example record returned (one entry in the list, one per PDF page):
load_pdfs__example = {
    "doc_id": "bitcoin_whitepaper",
    "source": "data/raw/bitcoin_whitepaper.pdf",
    "text": "Abstract. A purely peer-to-peer version of electronic cash...",
    "page": 1,
}


# ── src/chunker.py ──────────────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> "list[str]":
    """Sliding token window, decoded back to text chunks."""
    ...

# example return value (list of plain strings):
chunk_fixed__example = [
    "# Chroma\n\nthe AI-native open-source embedding database. Chroma gives you...",
    "...embedding database. Chroma gives you the tools to store embeddings and...",
]


def chunk_recursive(text: str, chunk_size: int = 500, separators: "list[str]" = None) -> "list[str]":
    """Paragraph-aware split; defaults: \\n\\n, \\n, '. ', ''. Falls back to raw
    tokens, merges small pieces back up toward chunk_size."""
    ...

# example return value (same shape as chunk_fixed, just split on paragraphs instead):
chunk_recursive__example = [
    "# Chroma\n\nthe AI-native open-source embedding database.",
    "Chroma gives you the tools to store embeddings and metadata...",
]


def chunk_document(
    record: dict,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap: int = 50,
) -> "list[dict]":
    """One chunk dict per chunk of record['text']."""
    ...

# example record returned (one entry in the list):
chunk_document__example = {
    "chunk_id": "a1b2c3d4e5f6...9f0a",  # sha256 hex of source|page|chunk_text
    "doc_id": "chroma_README",
    "source": "data/raw/chroma_README.md",
    "page": None,
    "chunk_index": 0,
    "char_start": 0,
    "text": "# Chroma\n\nthe AI-native open-source embedding database...",
}


# ── src/embedder.py ──────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        ...

    def embed(self, texts: "list[str]") -> "list[list[float]]":
        """Normalized embedding vectors, plain python floats, same order as input."""
        ...

# example return value (list of vectors, one per input text; all-MiniLM-L6-v2 -> 384 dims):
embed__example = [
    [0.0123, -0.0456, 0.0789, "... 384 floats total, L2-normalized (unit length) ..."],
    [-0.0211, 0.0033, 0.0501, "... another 384-float vector for the 2nd text ..."],
]


# ── src/vector_store.py ──────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, path: str = "data/chroma_db", collection_name: str = "personal_kb"):
        """Opens/creates a persistent Chroma collection with cosine space."""
        ...

    def upsert(
        self,
        ids: "list[str]",
        embeddings: "list[list[float]]",
        documents: "list[str]",
        metadatas: "list[dict]",
    ) -> None:
        """All four lists must be same length, same order. Writes to Chroma."""
        ...

    def query(self, query_embeddings: "list[list[float]]", n_results: int = 5) -> dict:
        """Returns Chroma's raw result dict (ids, documents, metadatas, distances
        - each a list of lists, one inner list per query embedding)."""
        ...

# example return value for ONE query embedding, n_results=2
# (outer list always has len == number of query embeddings passed in -
#  we only ever pass 1 at a time, so index [0] everywhere when using this)
query__example = {
    "ids": [["a1b2c3...", "f9e8d7..."]],
    "documents": [[
        "# Chroma\n\nthe AI-native open-source embedding database...",
        "Chroma gives you the tools to store embeddings and metadata...",
    ]],
    "metadatas": [[
        {"doc_id": "chroma_README", "source": "data/raw/chroma_README.md", "chunk_index": 0},
        {"doc_id": "chroma_README", "source": "data/raw/chroma_README.md", "chunk_index": 1},
    ]],
    "distances": [[0.1234, 0.2891]],  # cosine DISTANCE, not similarity - similarity = 1 - distance
}

    def count(self) -> int:
        """Total chunks currently stored in the collection."""
        ...


# ── ingest.py ────────────────────────────────────────────────────────────────

def main() -> None:
    """Loads data/raw -> chunks -> embeds -> upserts into Chroma, prints progress.
    Step 11 in progress: will add hash-based skip + stale-chunk cleanup, not yet wired.
    """
    ...


# ── NOT YET IMPLEMENTED ───────────────────────────────────────────────────────
# src/retriever.py   - retrieve(query, k) -> chunks + similarity scores, threshold, --debug, where= filter
# src/generator.py   - prompt assembly + gpt-4o-mini call, token budget, error handling
# src/logger.py      - ConversationLogger reuse/extension (chunk ids, scores, latency split)
# ask.py             - interactive CLI Q&A loop
