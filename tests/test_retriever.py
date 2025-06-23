# import pytest
from rag.retriever import retrieve_relevant_docs, load_retrieval_index
from langchain_core.documents import Document

def test_load_index_success():
    index = load_retrieval_index()
    assert index is not None
    assert hasattr(index, "similarity_search")

def test_retrieve_top_k():
    query = "What is Azure Blob Storage?"
    docs = retrieve_relevant_docs(query, top_k=3)
    assert isinstance(docs, list)
    assert len(docs) <= 3
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content
