# Conversational FAQ Chatbot

A CLI chatbot that answers Data Science Bootcamp FAQs. It keeps the full conversation history across turns, stays in scope using a guarded system prompt, tracks token usage against the model's context window, and logs every turn to a JSON file for audit/replay.

## How it works

```
User input → conversation history (messages[]) → OpenAI API (gpt-4o-mini)
                     ↑                                    ↓
             token counter                          assistant reply
                     ↑                                    ↓
             scope guard check ←── system prompt + few-shot examples
                                                          ↓
                                                  JSON logger (per turn)
```

Each turn appends to a single `messages` list that gets resent on every call — the model itself is stateless, so "memory" is just replaying history back to it. The system prompt carries the few-shot examples and the scope guard, so style and refusal behavior live in one place instead of being hardcoded into the loop. `token_utils.py` counts tokens before each call and warns at 80% of the context window, ahead of the request rather than after. `logger.py` writes a `user` and an `assistant` entry per turn straight to disk, so a crash mid-session doesn't lose anything already said.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```
OPENAI_API_KEY=your-key-here
```

Run it:

```bash
python chatbot.py
```

Type `exit` or `quit` to end the session and save the log.

## Sample conversation

```
==================================================
 Data Science Bootcamp — FAQ Chatbot
 Type 'exit' or 'quit' to end.
==================================================

You: What programming languages do I need to know?

Bot: Python is the primary programming language used in the bootcamp. Basic SQL
knowledge is also beneficial, but it will be taught as part of the curriculum.
[Tokens used: 213]

You: How long does the bootcamp take?

Bot: The self-paced program spans 6 months across 7 phases, covering topics such
as Machine Learning, Deep Learning, and Generative AI.
[Tokens used: 257]

You: What tools do I learn in Phase 3?

Bot: In Phase 3, you will learn tools such as Pandas for data manipulation,
Matplotlib and Seaborn for data visualization, and Scikit-learn for machine
learning. This phase focuses on data analysis and model building techniques.
[Tokens used: 322]

You: Is there a capstone project?

Bot: Yes, there is a capstone project included in the bootcamp. It allows you to
apply the skills and knowledge you've gained throughout the program to a
real-world data science problem.
[Tokens used: 373]

You: Can you help me with my taxes?

Bot: I can only answer questions about the Data Science Bootcamp curriculum.
[Tokens used: 455]

You: exit

Session saved → logs/session_20261017_221014_f9b38d0a.json
   Turns: 20 | Tokens used: 582
Goodbye!
```

## Token tracking

The running token count prints after every turn:

```
[Tokens used: 213]
[Tokens used: 257]
[Tokens used: 322]
[Tokens used: 373]
[Tokens used: 455]
```

When usage crosses 80% of `CONTEXT_LIMIT` (128,000 tokens for `gpt-4o-mini`), a warning prints before the API call:

```
Warning: 102,412 / 128,000 tokens used (80%)
```

## Known limitations

- No web UI — command-line only
- No persistent memory across sessions — each run starts a fresh conversation
- Single-user only, no concurrency handling
- Scope guard is tuned conservative — it occasionally refuses borderline but arguably on-topic questions (e.g. comparing the bootcamp to a university degree). Could loosen this with a couple more few-shot examples if it turns out too strict in practice
