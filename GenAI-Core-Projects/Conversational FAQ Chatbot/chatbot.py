from openai import OpenAI
from prompts import SYSTEM_PROMPT
from token_utils import count_tokens, WARNING_THRESHOLD, CONTEXT_LIMIT
from logger import ConversationLogger

import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    # reads OPENAI_API_KEY from .env via python-dotenv
    api_key=os.getenv("OPENAI_API_KEY"),
)

def chat():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    logger = ConversationLogger(model="gpt-4o-mini")

    print("=" * 50)
    print(" Data Science Bootcamp — FAQ Chatbot")
    print(" Type 'exit' or 'quit' to end.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Please enter a question.\n")
            continue
        if user_input.lower() in ("exit", "quit"):
            logger.save()
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # check before the call so we warn ahead of a request that might get rejected
        tokens = count_tokens(messages)
        if tokens > CONTEXT_LIMIT * WARNING_THRESHOLD:
            print(f"Warning: {tokens:,} / {CONTEXT_LIMIT:,} tokens used ({tokens/CONTEXT_LIMIT:.0%})")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,   # low — FAQ answers should be consistent
                max_tokens=300,
            )
        except Exception as e:
            print(f"\nSomething went wrong talking to the API: {e}\n")
            messages.pop()  # drop the user turn that never got a reply
            continue

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"\nBot: {reply}")
        print(f"[Tokens used: {count_tokens(messages):,}]\n")

        # two entries per turn — keeps the log append-only instead of buffering a half-built turn
        logger.log_turn("user", user_input, tokens)
        logger.log_turn("assistant", reply, count_tokens(messages))

if __name__ == "__main__":
    chat()