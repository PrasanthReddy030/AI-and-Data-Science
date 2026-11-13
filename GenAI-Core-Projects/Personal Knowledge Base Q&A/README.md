# Personal Knowledge Base Q&A

A command-line RAG pipeline over a personal document collection. Documents are chunked,
embedded locally, and indexed in ChromaDB; questions are answered by gpt-4o-mini
grounded strictly in retrieved context, with source citations.

Built from scratch (loader, chunker, embedder, vector store, retriever, grounded
generation) without an orchestration framework like LangChain - the one exception is
`chunk_recursive`, which uses LangChain's `RecursiveCharacterTextSplitter` (see
[Design Decisions](#design-decisions)).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Drop source documents (`.txt`, `.md`, `.pdf`) into `data/raw/`, then build the index:

```bash
python ingest.py
```

Re-running `ingest.py` is safe - it hashes each file and only re-chunks/re-embeds
files that are new or changed. Deleted files have their chunks removed automatically.

Ask questions:

```bash
python ask.py
python ask.py --debug          # print retrieved chunks and scores
python ask.py --k 8 --threshold 0.25
```

## Architecture

**Ingest path** (`ingest.py`):

```
data/raw/*.{txt,md,pdf}
        |
        v
  loader.py  ---------- per-file hash check (skip unchanged)
        |
        v
  chunker.py  ---------- fixed-size (token sliding window) or
        |                 recursive (LangChain, paragraph-aware)
        v
  embedder.py  --------- sentence-transformers, batched, normalized
        |
        v
  vector_store.py  ----- ChromaDB PersistentClient, cosine space
                          (stale chunks deleted first on re-ingest)
```

**Query path** (`ask.py`):

```
  question
        |
        v
  retriever.py  -------- embed query -> Chroma top-k -> distance to
        |                 similarity -> threshold filter
        |
        +---- nothing clears threshold --> "not found", no LLM call
        |
        v
  generator.py  -------- build_context() enforces token budget,
        |                 drops lowest-scoring chunks first
        v
  gpt-4o-mini  ---------- grounded system prompt, citations,
        |                 "not found" fallback on insufficient context
        v
  logger.py  ------------ chunk IDs, scores, retrieval/LLM latency split
```

## Design Decisions

- **Chunk ID = sha256(source | page | chunk_text).** Deterministic IDs let `ingest.py`
  use Chroma's `upsert` for idempotency - re-running on an unchanged file is a no-op,
  and a changed file's old chunks are deleted by `source` before the new ones are
  written. `page` is included in the hash because a PDF's pages share one `source`
  path; without it, identical boilerplate text on two pages would collide.

- **Similarity threshold before the LLM call.** `retriever.py` converts Chroma's
  cosine distance to similarity (`1 - distance`) and drops anything below 0.3 by
  default. If nothing clears the bar, `ask.py` returns "I couldn't find this in your
  notes" without spending an API call - this is also what keeps the two out-of-corpus
  control questions in the eval set from reaching the LLM at all.

- **Context budget drops the lowest-scoring chunks first, not the smallest.** Chunks
  arrive from Chroma already ranked by relevance; `build_context()` adds them in that
  order and stops once the next one would exceed the token budget, rather than
  backfilling with a smaller low-ranked chunk that happens to fit. One edge case this
  doesn't handle gracefully: if the single highest-scoring chunk alone exceeds the
  budget, the context comes back empty and the system answers "not found" - which is
  technically true but misleading, since something relevant *was* retrieved. Flagged
  in [Known Limitations](#known-limitations) rather than fixed, since it never
  triggered in practice with this corpus's chunk sizes.

- **`chunk_recursive` uses LangChain; everything else doesn't.** The original
  hand-rolled implementation split on paragraph breaks with no overlap between
  chunks. It was replaced with LangChain's `RecursiveCharacterTextSplitter` to get
  real token overlap on the recursive strategy, matching the overlap behavior
  `chunk_fixed` already had. This is a deliberate, scoped use of one library
  component - the loader, fixed chunking, embedder, vector store, retriever, and
  generator have no LangChain dependency.

- **Metadata filtering.** `VectorStore.query()` and `Retriever.retrieve()` both accept
  a `where` filter, forwarded directly to Chroma (e.g. `where={"source": "faiss_README.md"}`
  restricts retrieval to a single document). Demonstrated in `tests/test_retriever.py::test_retrieve_passes_query_parameters_to_store`.

## Corpus

27 real public documents (not personal notes) - 17 `.md`, 4 `.txt`, 6 `.pdf` - spanning
ML/AI tooling READMEs (Faiss, Chroma, FastAPI, MLflow, PyTorch, Transformers, ...),
Python PEPs, Google engineering-practices docs, and distributed-systems papers (GFS,
Dynamo, Raft, the Bitcoin whitepaper). Chosen to be material the author already
understands well enough to judge retrieval and generation quality by hand.

## Experiments

22 eval questions (`eval/test_questions.json`): 15 straightforward lookups across the
corpus's sub-clusters, 5 deliberately hard/paraphrased or ambiguous questions, and 2
out-of-corpus control questions excluded from hit-rate scoring. Hit rate = fraction of
in-corpus questions where an expected source appears in the top-k retrieved chunks.

| Chunk size | Overlap | Strategy  | Top-k | Hit rate |
|-----------:|--------:|-----------|------:|---------:|
| 500        | 50      | fixed     | 4     | 100% (baseline) |
| 200        | 50      | fixed     | 4     | 95% |
| 1000       | 50      | fixed     | 4     | 100% |
| 500        | 0       | fixed     | 4     | 95% |
| 500        | 100     | fixed     | 4     | 100% |
| 500        | 50      | fixed     | 2     | 95% |
| 500        | 50      | fixed     | 8     | 100% |
| 500        | 50      | recursive | 4     | 100% |

Full per-question results in `data/experiments/experiment_results.json`.

Manual grading of the 20 answerable questions (`eval/grading_results.json`): **20
correct, 2 partial, 0 hallucinated.** Both partial verdicts were retrieval failures,
not generation failures - in each case the model correctly refused to answer rather
than fabricate a response when the right chunks weren't in its context (see Failure
Cases below).

## Failure Cases

1. **Ambiguous cross-document questions lose at low top-k.** A question deliberately
   written to match either of two sources ("How does a system decide which replica's
   version of the data should win...", expecting Dynamo or Raft) only failed at
   `k=2`; it succeeded from `k=4` onward. At `k=2` only one candidate source made it
   into the window, and it happened to be the wrong one. Fix: raise `k` for
   open-ended questions, or add query rewriting/decomposition so an ambiguous
   question is split into sub-queries before retrieval.

2. **Topically adjacent sibling documents blur at small chunk sizes.** "What
   technique involves showing a language model a few example input-output pairs..."
   (expecting `prompt_eng_guide_basics.md`) retrieved `prompt_eng_guide_intro.md`
   instead at `chunk_size=200` and at `overlap=0`. The two documents are part of the
   same guide series with overlapping vocabulary; smaller or non-overlapping chunks
   carry less surrounding context, making it easier for a chunk from the sibling
   document to score as relevant. Fix: keep overlap non-zero for this corpus, or
   add a re-ranking step that considers full-document context, not just the chunk.

3. **Eval question labels can be wrong even when retrieval is right.** One eval
   question originally assumed two prompt-engineering guides covered document
   chunking; they don't - that content isn't in the corpus. Every retrieval config
   "failed" that question until the expected source was corrected to a real match
   (few-shot prompting, which the same document does cover). This wasn't a pipeline
   bug: it was a reminder that a hit-rate number is only as trustworthy as the
   ground truth it's measured against, and ground truth should be verified against
   the actual corpus text before trusting a miss as a pipeline failure.

## Known Limitations

- No re-ranking or hybrid (keyword + semantic) search - purely dense vector retrieval,
  which is the main driver behind failure case 1.
- `build_context()` returns an empty string if the single highest-scoring chunk alone
  exceeds the token budget, producing a "not found" answer that's misleading about
  *why* nothing was found. Never triggered with this corpus's configs, but not
  handled.
- The corpus is public documentation and papers, not the author's personal notes -
  chosen deliberately so retrieval and generation quality could be verified against
  independently known facts, at the cost of not testing retrieval against the kind of
  informal, inconsistent writing personal notes usually contain.
- No multi-turn conversation memory - each question in `ask.py` is retrieved and
  answered independently, with no reference to prior turns in the session.
- `_interface_map.py` and `experiments.py` / `grade_answers.py` are development/
  analysis scaffolding, not part of the shipped pipeline (`ingest.py` + `ask.py` +
  `src/`).

## Tests

```bash
pytest tests/ -v
```

Covers chunker (overlap correctness, no empty chunks, both strategies), loader (bad
file handling, unsupported extensions, PDF page extraction), retriever (threshold
filtering, distance-to-similarity conversion, metadata filter passthrough), and one
mocked end-to-end test (real chunking/retrieval logic, OpenAI call stubbed).
