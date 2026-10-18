import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

class ConversationLogger:
    """
    One instance per chatbot session.
    Creates a unique log file on init; appends turns in real time.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        LOG_DIR.mkdir(exist_ok=True)

        self.model: str = model
        self.session_id: str = str(uuid.uuid4())[:8]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath: Path = LOG_DIR / f"session_{ts}_{self.session_id}.json"

        self._turn_count: int = 0
        self._last_token_count: int = 0

        # write immediately so a crash before the first turn still leaves a valid file
        self._write_header()

    def _write_header(self):
        """Create the file with metadata; turns will be appended below."""
        header = {
            "session_id": self.session_id,
            "model": self.model,
            "started_at": datetime.now().isoformat(),
            "turns": []          # turns array — appended to in log_turn()
        }
        with open(self.filepath, "w") as f:
            json.dump(header, f, indent=2)


    def log_turn(self, role: str, content: str, tokens_so_far: int) -> None:
        """
        Append one turn and rewrite the entire file immediately.
        Called after EVERY user message and EVERY assistant reply.

        Args:
            role          : "user" or "assistant"
            content       : the raw message text
            tokens_so_far : cumulative token count AFTER this turn
        """
        self._turn_count += 1
        self._last_token_count = tokens_so_far

        entry = {
            "turn":          self._turn_count,
            "timestamp":     datetime.now().isoformat(),
            "role":          role,
            "content":       content,
            "tokens_so_far": tokens_so_far
        }

        # full read-modify-write per call, not append-only — keeps one valid JSON
        # object on disk at all times instead of a JSONL stream. fine at chatbot scale,
        # would need to switch to line-delimited if sessions got into the thousands of turns
        with open(self.filepath, "r") as f:
            data = json.load(f)

        data["turns"].append(entry)

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)


    def save(self) -> None:
        """
        Write final summary fields to the log file and close out the session.
        Call this once when the conversation ends.
        """
        with open(self.filepath, "r") as f:
            data = json.load(f)

        data["ended_at"]          = datetime.now().isoformat()
        data["total_turns"]       = self._turn_count
        data["total_tokens_used"] = self._last_token_count

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSession saved → {self.filepath}")
        print(f"   Turns: {self._turn_count} | Tokens used: {self._last_token_count:,}")