from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
    """
    Returns a LangChain-compatible embedding model wrapper.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def embed_text(text: str) -> List[float]:
    """
    Embeds a single query using BGE model (with 'query:' prefix).
    """
    embedder = get_embedder()
    formatted = "query: " + text.strip()
    embedding = embedder.embed_query(formatted)
    return embedding


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of queries using BGE model.
    """
    embedder = get_embedder()
    formatted_texts = ["query: " + text.strip() for text in texts]
    embeddings = embedder.embed_documents(formatted_texts)
    return embeddings #[emb.tolist() for emb in embeddings]


if __name__ == "__main__":
    # Example usage
    # Test embedding a single text
    sample_text = "What is the capital of France?"
    embedding = embed_text(sample_text)
    print(f"Text: {sample_text}\nEmbedding: {embedding[:5]}... (length: {len(embedding)})\n")

    # Uncomment the following lines to test embedding multiple texts
    # sample_texts = ["What is the capital of France?", "How does semantic caching work?"]
    # embeddings = embed_texts(sample_texts)
    # for text, embedding in zip(sample_texts, embeddings):
    #     print(f"Text: {text}\nEmbedding: {embedding[:5]}... (length: {len(embedding)})\n")