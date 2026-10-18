import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o-mini")-> int:
    """Count total tokens in the current conversation history."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4 # per-message overhead
        total += len(enc.encode(msg["content"]))
    return total

CONTEXT_LIMIT = 128_000   # gpt-4o-mini context window
WARNING_THRESHOLD = 0.80  # warn at 80% usage
