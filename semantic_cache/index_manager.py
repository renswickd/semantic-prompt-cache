import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from logger.logger import get_logger
from semantic_cache.embedder import get_embedder
from pathlib import Path


logger = get_logger(__name__)

# Paths
DEFAULT_INDEX_DIR = "data/cache_store"
DEFAULT_INDEX_NAME = "semantic_cache"

def create_faiss_index(
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME
) -> FAISS:
    """
    Create a new FAISS index with given texts and optional metadata.
    Saves the index to disk.
    """
    embedder = get_embedder()

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas
    )
    logger.info(f"[index_manager] Created FAISS index")

    save_faiss_index(vector_store, index_dir, index_name)
    logger.info(f"[index_manager] FAISS index saved to {index_dir}/{index_name}.faiss")

    return vector_store


def load_faiss_index(
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME
) -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.
    Returns None if not found.
    """
    
    embedder = get_embedder()

    index_path = Path(index_dir) / f"{index_name}.faiss"
    if not index_path.exists():
        logger.warning(f"[index_manager] Index not found at {index_path}")
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        logger.info(f"[index_manager] Loaded FAISS index from {index_path}")

        return vector_store
    
    except Exception as e:
        logger.error(f"[index_manager] Failed to load FAISS index: {e}")
        return None


def save_faiss_index(
    vector_store: FAISS,
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME
):
    """
    Saves the FAISS index to disk.
    """
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(folder_path=index_dir, index_name=index_name)
    # print(f"[index_manager] FAISS index saved to {index_dir}/{index_name}.faiss")
    logger.info(f"[index_manager] FAISS index saved to {index_dir}/{index_name}.faiss")


def reset_faiss_index(index_dir: str = DEFAULT_INDEX_DIR):
    """
    Deletes all FAISS index files (use with caution).
    """
    for file in Path(index_dir).glob("*"):
        file.unlink()
    # print(f"[index_manager] Reset FAISS index in {index_dir}")
    logger.info(f"[index_manager] Reset FAISS index in {index_dir}")


if __name__ == "__main__":
    sample_texts = ["What is the capital of France?", "How does semantic caching work?"]
    sample_metadatas = [{"source": "doc1"}, {"source": "doc2"}]

    # Create index
    index = create_faiss_index(sample_texts, sample_metadatas)
    # print(dir(index))
    print(f"Index created with {len(index.docstore._dict)} documents.")

    # # Load index
    # loaded_index = load_faiss_index()

    # # Reset index
    # reset_faiss_index()