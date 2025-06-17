# semantic_cache/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import os

# Global variable to hold the model instance
_model = None

def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
    """
    Loads the BGE embedding model and returns the model instance.
    Uses singleton pattern to avoid reloading.
    """
    global _model
    if _model is None:
        print(f"[embedder] Loading model '{model_name}' on {device} ...")
        _model = SentenceTransformer(model_name)
        _model = _model.to(device)
    return _model


def embed_text(text: str) -> List[float]:
    """
    Embeds a single query using BGE model (with 'query:' prefix).
    """
    embedder = get_embedder()
    formatted = "query: " + text.strip()
    embedding = embedder.encode(formatted, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of queries using BGE model.
    """
    embedder = get_embedder()
    formatted_texts = ["query: " + text.strip() for text in texts]
    embeddings = embedder.encode(formatted_texts, normalize_embeddings=True)
    return [emb.tolist() for emb in embeddings]


if __name__ == "__main__":
    # Example usage
    sample_texts = ["What is the capital of France?", "How does semantic caching work?"]
    embeddings = embed_texts(sample_texts)
    for text, embedding in zip(sample_texts, embeddings):
        print(f"Text: {text}\nEmbedding: {embedding[:5]}... (length: {len(embedding)})\n")