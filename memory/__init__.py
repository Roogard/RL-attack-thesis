from memory.full_history import FullHistoryMemory
from memory.rag import RAGMemory
from memory.rl_memory import RLMemory

MEMORY_REGISTRY = {
    "full_history": FullHistoryMemory,
    "rag": RAGMemory,
    "rl_memory": RLMemory,
    # "kv_store": KVStoreMemory,
    # "temporal_rag": TemporalRAGMemory,
    # "graph_memory": GraphMemory,
}
