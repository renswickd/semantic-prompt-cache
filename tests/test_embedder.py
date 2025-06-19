from semantic_cache.embedder import embed_text, embed_texts

def test_embed_text_returns_vector():
    query = "best places to visit in Spain"
    embedding = embed_text(query)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

def test_embed_texts_returns_multiple_vectors():
    queries = ["hello world", "best beaches in Europe"]
    embeddings = embed_texts(queries)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)