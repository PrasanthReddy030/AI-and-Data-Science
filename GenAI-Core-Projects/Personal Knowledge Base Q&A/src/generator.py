"""Grounds gpt-4o-mini to answer only from retrieved context, with citations, and
builds the context block that stays under a token budget before the call is made."""

SYSTEM_PROMPT = """\
You are a question-answering assistant for a personal knowledge base.

Your task is to answer the user's question using ONLY the information
provided in the retrieved context.

STRICT RULES:
1. Use only the provided context to answer questions.
2. NEVER use outside knowledge, pretrained knowledge, assumptions,
   guesses, or information not explicitly supported by the context.
3. If the context contains no information relevant to answering the
   question, respond with exactly:
   "I couldn't find this in your notes."
4. If the context contains some relevant information but does not
   fully answer the question, provide the supported partial answer.
   Clearly identify what information is available and what could
   not be found in the context.
5. Never attempt to fill missing information using your own knowledge.
6. If the context contains conflicting information, explicitly
   identify the disagreement rather than choosing one version.

CITATION RULES:
1. Every factual claim must have a citation.
2. Use the following citation formats:
   - PDF sources: [source, p.N]
   - TXT/Markdown sources: [source, p.N/A]
3. Use the exact source filename provided in the context.
4. Use the exact page number provided in the context.
5. Never invent, estimate, or assume page numbers.
6. If page information is missing or None, use p.N/A.
7. Place citations immediately after the claims they support.
8. If multiple sources support a claim, cite each relevant source.

RESPONSE STYLE:
1. Be clear, concise, and directly answer the question.
2. Do not introduce information that is not supported by the context.
3. Do not fabricate citations or supporting evidence.
4. When providing a partial answer, clearly distinguish supported
   information from information that could not be found.
5. Prioritize accuracy and faithfulness to the retrieved context
   over completeness.

Remember: You are answering questions about the user's notes,
not answering from your general knowledge.
"""

import tiktoken


def build_context(chunks: list[dict], max_tokens: int = 3000) -> str:
    """
    Format retrieved chunks into numbered, source-labeled blocks, stopping once
    the running token count would exceed max_tokens.

    Chunks must already be sorted by relevance score (highest first) - once a
    chunk doesn't fit, every chunk after it is dropped too, since they're all
    lower-scoring. This keeps the kept chunks' relevance ordering intact rather
    than cherry-picking smaller low-priority chunks that happen to fit.

    Returns the formatted context string, or "" if even the first chunk alone
    exceeds the budget.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    context_blocks = []
    total_tokens = 0

    for chunk in chunks:

        page = chunk.get("page")

        if page is None:
            page = "N/A"

        block = (
            f"[{len(context_blocks) + 1}] "
            f"Source: {chunk['source'].split('/')[-1]}, "
            f"Page: {page}\n"
            f"{chunk['text']}"
        )

        # separator counts toward the budget too, or the real prompt sent to
        # the LLM ends up slightly over max_tokens
        separator = "\n\n" if context_blocks else ""
        block_tokens = len(enc.encode(separator + block))

        if total_tokens + block_tokens > max_tokens:
            break

        context_blocks.append(block)
        total_tokens += block_tokens

    # chunks existed but the first (highest-scoring) one alone didn't fit -
    # silent empty context would make the LLM say "not found" for the wrong reason
    if chunks and not context_blocks:
        print(
            "WARNING: Retrieved chunks exceeded the "
            "context token budget. No chunks were added."
        )

    return "\n\n".join(context_blocks)


import os

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using the retrieved context and system prompt.

    Returns:
        Generated answer or a readable error message.
    """

    # short-circuit before spending an API call - also a backstop in case a
    # caller skips the retriever's own threshold check
    if not context.strip():
        return "I couldn't find this in your notes."

    user_message = f"""\
Use the following retrieved context to answer the question.

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>

Remember to follow the system instructions and cite every factual claim.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )

        answer = response.choices[0].message.content

        if not answer:
            return "Sorry, the model returned an empty response. Please try again."

        return answer.strip()

    except (RateLimitError, APITimeoutError, APIError) as e:
        print(f"OpenAI API error: {e}")

        return (
            "Sorry, I couldn't generate an answer because the "
            "OpenAI API is temporarily unavailable. Please try again."
        )
