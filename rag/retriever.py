from typing import List, Dict, Optional
from langchain_core.documents import Document
from semantic_cache.embedder import get_embedder
from langchain_community.vectorstores import FAISS
from logger.logger import get_logger
from pathlib import Path
import os

logger = get_logger(__name__)

# Constants
RETRIEVAL_INDEX_DIR = "data/vector_store"
RETRIEVAL_INDEX_NAME = "azure_docs_index"
TOP_K = 2


def load_retrieval_index(index_dir: str = RETRIEVAL_INDEX_DIR, index_name: str = RETRIEVAL_INDEX_NAME) -> Optional[FAISS]:
    """
    Load FAISS index containing the knowledge base documents.
    """
    embedder = get_embedder()
    index_path = Path(index_dir) / f"{index_name}.faiss"

    if not index_path.exists():
        logger.warning(f"[retriever] Retrieval index not found at {index_path}")
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        logger.info(f"[retriever] Loaded vector index from {index_path}")
        return vector_store
    except Exception as e:
        logger.error(f"[retriever] Failed to load vector index: {e}")
        return None


def retrieve_relevant_docs(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    Retrieve top-K most relevant documents for the query.
    """
    vector_store = load_retrieval_index()
    if not vector_store:
        logger.warning("[retriever] No retrieval index loaded. Returning empty list.")
        return []

    try:
        results = vector_store.similarity_search(query, k=top_k)
        logger.info(f"[retriever] Retrieved {len(results)} documents for query.")
        return results
    except Exception as e:
        logger.error(f"[retriever] Retrieval error: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    example_query = "What are the key features of Azure Cognitive Services?"
    retrieved_docs = retrieve_relevant_docs(example_query)
    if retrieved_docs:
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            print(f"- {doc.metadata.get('title', 'No title')}: {doc.page_content[:100]}...")
    else:
        print("No relevant documents found.")