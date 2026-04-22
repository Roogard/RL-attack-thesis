from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """Interface for all memory architectures.

    Each implementation handles indexing (ingesting conversation history)
    and retrieval (finding relevant context for a question).
    The reading stage (LLM call) is handled by harness.py.
    """

    @abstractmethod
    def index(self, sessions, dates, session_ids):
        """Ingest conversation history for one question's haystack.

        Called once per question before retrieval.

        Args:
            sessions: list of sessions, each a list of {role, content} turns
            dates: list of date strings, one per session
            session_ids: list of session identifier strings
        """
        pass

    @abstractmethod
    def retrieve(self, question, question_date, top_k=10):
        """Return context string relevant to the question.

        Args:
            question: the user's question string
            question_date: date string for temporal reasoning
            top_k: max number of chunks/facts to retrieve

        Returns:
            str: formatted context to insert into the prompt
        """
        pass

    @abstractmethod
    def clear(self):
        """Reset the store between questions."""
        pass

    @classmethod
    def index_batch(cls, instances, questions):
        """Index B questions in parallel.

        Default implementation just loops sequentially. Memory types whose
        index() does heavy LLM work (RLMemory) override this to actually
        batch generate() calls across questions, which is what saturates
        the GPU under continuous batching.

        Args:
            instances: list of B already-cleared MemoryStore instances
            questions: list of B question dicts (LongMemEval items)
        """
        for inst, q in zip(instances, questions):
            inst.index(
                q["haystack_sessions"],
                q["haystack_dates"],
                q.get(
                    "haystack_session_ids",
                    [str(i) for i in range(len(q["haystack_sessions"]))],
                ),
            )
