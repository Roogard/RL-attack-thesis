from memory.base import MemoryStore


class FullHistoryMemory(MemoryStore):
    """In-context baseline: returns the entire conversation history as context.

    No indexing or retrieval logic — just formats all sessions into a string.
    This is the upper bound for context availability (but hits token limits).
    """

    def __init__(self):
        self.history = ""

    def index(self, sessions, dates, session_ids):
        chunks = []
        for date, session in zip(dates, sessions):
            turns = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content'].strip()}"
                for turn in session
            )
            chunks.append(f"[Session Date: {date}]\n{turns}")
        self.history = "\n\n---\n\n".join(chunks)

    def retrieve(self, question, question_date, top_k=10):
        return self.history

    def clear(self):
        self.history = ""
