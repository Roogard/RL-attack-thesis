from memory.full_history import FullHistoryMemory
from memory.rag import RAGMemory

MEMORY_REGISTRY = {
    "full_history": FullHistoryMemory,
    "rag": RAGMemory,
    # "kv_store": KVStoreMemory,
    # "temporal_rag": TemporalRAGMemory,
    # "rl_memory": RLMemory,
    # "graph_memory": GraphMemory,
}
