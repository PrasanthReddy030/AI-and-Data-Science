"""Session logger adapted from the Week 9 chatbot's ConversationLogger - extended here
to record retrieved chunk IDs/scores and a retrieval-vs-LLM latency split per turn."""

import json
import uuid

from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class ConversationLogger:
    """
    One instance per RAG session.

    Creates a unique JSON log file on initialization.
    Records each question-answer interaction, including
    retrieval metadata and timing information.
    """

    def __init__(self, model: str = "gpt-4o-mini"):

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.model: str = model
        self.session_id: str = str(uuid.uuid4())[:8]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filepath: Path = (
            LOG_DIR / f"session_{ts}_{self.session_id}.json"
        )

        self._turn_count: int = 0
        self._last_turn: dict | None = None

        # Write immediately so the session file exists even
        # if the program exits before the first question.
        self._write_header()

    def _write_header(self):
        """Create the initial session metadata."""

        header = {
            "session_id": self.session_id,
            "model": self.model,
            "started_at": datetime.now().isoformat(),
            "turns": []
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(header, f, indent=2)

    def log_qa_turn(
        self,
        question: str,
        answer: str,
        chunks: list[dict],
        retrieval_ms: float,
        llm_ms: float,
    ) -> None:
        """
        Record one complete RAG question-answer interaction.

        Args:
            question: Original user question.
            answer: Generated answer or fallback message.
            chunks: Retrieved chunks and their metadata.
            retrieval_ms: Time spent retrieving chunks.
            llm_ms: Time spent generating the answer.
        """

        self._turn_count += 1

        # chunk text isn't stored - keeps the log small, and chunk_id is enough
        # to look up the full text in Chroma if needed later
        retrieved_chunks = []

        for chunk in chunks:

            retrieved_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "source": chunk.get("source"),
                "page": chunk.get("page"),
            })

        entry = {
            "turn": self._turn_count,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_ms": round(retrieval_ms, 2),
            "llm_ms": round(llm_ms, 2),
        }

        self._last_turn = entry

        # full read-modify-write per turn, not an append-only log - keeps one
        # valid JSON object on disk at all times instead of a JSONL stream.
        # fine at this session scale; would need line-delimited writes if a
        # session ever ran into the thousands of turns
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["turns"].append(entry)

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save(self) -> None:
        """
        Write final session summary fields.
        Call once when the session ends.
        """

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["ended_at"] = datetime.now().isoformat()
        data["total_turns"] = self._turn_count

        data["total_retrieval_ms"] = round(
            sum(turn["retrieval_ms"] for turn in data["turns"]),
            2
        )

        data["total_llm_ms"] = round(
            sum(turn["llm_ms"] for turn in data["turns"]),
            2
        )

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"\nSession saved -> {self.filepath}")
        print(f"   Turns: {self._turn_count}")
        print(f"   Retrieval time: {data['total_retrieval_ms']:.2f} ms")
        print(f"   LLM time: {data['total_llm_ms']:.2f} ms")
