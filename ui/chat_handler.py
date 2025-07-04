from rag.query_router import query_router
from semantic_cache.operations import clear_cache

def handle_query(query: str) -> str:
    """
    Process user query via RAG pipeline.
    """
    return query_router(query)

def clear_cache_handler():
    """
    Clear semantic cache store.
    """
    clear_cache()
