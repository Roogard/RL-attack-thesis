import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from memory.base import MemoryStore


class RAGMemory(MemoryStore):
    """Vector DB memory using ChromaDB with round-level chunking.

    Each user-assistant turn pair is embedded and stored as a separate chunk.
    Retrieval returns the top_k most semantically similar rounds, sorted
    chronologically for coherent context.

    Note: chromadb.Client() is a process-singleton, so when the harness
    creates B instances per batch we must give each a unique collection name
    to avoid name collisions.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._collection_name = f"memory_{uuid.uuid4().hex}"
        self._ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=self._collection_name, embedding_function=self._ef
        )

    def index(self, sessions, dates, session_ids):
        documents = []
        metadatas = []
        ids = []

        for session, date, sid in zip(sessions, dates, session_ids):
            # Pair consecutive turns into rounds (user + assistant)
            round_idx = 0
            i = 0
            while i < len(session):
                if i + 1 < len(session) and session[i]["role"] == "user" and session[i + 1]["role"] == "assistant":
                    text = (
                        f"User: {session[i]['content'].strip()}\n"
                        f"Assistant: {session[i + 1]['content'].strip()}"
                    )
                    i += 2
                else:
                    # Unpaired turn (trailing user message or unexpected order)
                    turn = session[i]
                    text = f"{turn['role'].capitalize()}: {turn['content'].strip()}"
                    i += 1

                documents.append(text)
                metadatas.append({"date": date, "session_id": sid, "round_index": round_idx})
                ids.append(f"{sid}_{date}_r{round_idx}")
                round_idx += 1

        # ChromaDB add in batches (avoid exceeding batch size limits)
        batch_size = 5000
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self._collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    def retrieve(self, question, question_date, top_k=10):
        results = self._collection.query(query_texts=[question], n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return ""

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Sort by date then round_index for chronological ordering
        paired = sorted(zip(metas, docs), key=lambda x: (x[0]["date"], x[0]["round_index"]))

        chunks = []
        for meta, doc in paired:
            chunks.append(f"[Session Date: {meta['date']}]\n{doc}")

        return "\n\n---\n\n".join(chunks)

    def index_raw_embeddings(self, embeddings, metadatas, ids, documents=None):
        """Inject pre-computed L2-normalized vectors directly into the collection.

        ChromaDB skips the embedding function for items where embeddings are
        provided. `documents` is the text returned by retrieve() when the
        vector wins; default is a neutral placeholder so retrieval-displacement
        is the only effect on accuracy. metadatas must include 'date' and
        'round_index' (used by retrieve() for chronological ordering).
        """
        if documents is None:
            documents = ["[injected]" for _ in ids]
        self._collection.add(
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
            ids=list(ids),
        )

    def clear(self):
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name, embedding_function=self._ef
        )

    def __del__(self):
        # Free chromadb's in-memory state for this instance — otherwise
        # the singleton client accumulates orphaned collections all run.
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
