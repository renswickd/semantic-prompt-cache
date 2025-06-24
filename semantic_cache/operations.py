from typing import Optional
from datetime import datetime
from uuid import uuid4
from langchain_core.documents import Document

from semantic_cache.index_manager import (
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
    reset_faiss_index,
)
from semantic_cache.embedder import embed_text
from logger.logger import get_logger

# Constants
CACHE_THRESHOLD = 0.50
DEFAULT_INDEX_DIR = "data/cache_store"
DEFAULT_INDEX_NAME = "semantic_cache"

logger = get_logger(__name__)


def get_from_cache(query: str) -> Optional[str]:
    """
    Search the FAISS index for a semantically similar cached response.
    Returns response if similarity is below threshold.
    """
    index = load_faiss_index(DEFAULT_INDEX_DIR, DEFAULT_INDEX_NAME)
    if not index:
        logger.warning("[cache] No index found. Cache miss.")
        return None

    try:
        result = index.similarity_search_with_score(query, k=1)
        # print(result)
        if result:
            doc, score = result[0]
            print()
            # print(doc, score)
            logger.info(f"[cache] Similarity score: {score}")
            if score >= CACHE_THRESHOLD:
                logger.info("[cache] Cache hit")
                return doc.metadata.get("response")
            else:
                logger.info("[cache] Cache miss (score too high)")
    except Exception as e:
        logger.error(f"[cache] Error during similarity search: {e}")

    return None


def set_in_cache(query: str, response: str):
    """
    Add a query-response pair to the FAISS index and persist it.
    """
    index = load_faiss_index(DEFAULT_INDEX_DIR, DEFAULT_INDEX_NAME)

    doc_id = str(uuid4())
    now = datetime.now()

    metadata = {
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "response": response,
        "id": doc_id,
        "hits": 1
    }

    try:
        if index:
            index.add_texts([query], metadatas=[metadata], ids=[doc_id])
            logger.info(f"[cache] Added to existing index with ID {doc_id}")
        else:
            index = create_faiss_index([query], [metadata])
            logger.info(f"[cache] Created new index with initial entry {doc_id}")

        save_faiss_index(index, DEFAULT_INDEX_DIR, DEFAULT_INDEX_NAME)

    except Exception as e:
        logger.error(f"[cache] Failed to store cache entry: {e}")


def clear_cache():
    """
    Clears all files in the cache directory.
    """
    reset_faiss_index(DEFAULT_INDEX_DIR)
    logger.info("[cache] Cleared all cache files")

if __name__ == "__main__":

    clear_cache() # Clear cache before testing
    query = "What are the uses of Machine Learning in healthcare?"
    cached = get_from_cache(query)
    if cached:
        print("✅ Cache Hit:", cached)
    else:
        print("🚫 Cache Miss — calling LLM...")
        response = "Rome, Venice, Florence..." 
        set_in_cache(query, response)
