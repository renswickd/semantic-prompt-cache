import pytest
from rag.response_generator import generate_response_text
from langchain_core.documents import Document

def test_response_generator_appends_sources():
    llm_output = "Azure Blob Storage is a scalable object store."
    docs = [
        Document(page_content="Some content here", metadata={"source": "https://azure.microsoft.com/storage"}),
        Document(page_content="Other content", metadata={"source": "https://learn.microsoft.com/blob"}),
    ]
    result = generate_response_text(llm_output, docs)

    assert "Azure Blob Storage" in result
    assert "Sources:" in result
    assert "azure.microsoft.com" in result
    assert "learn.microsoft.com" in result

def test_response_generator_without_sources():
    llm_output = "Azure offers many storage options."
    docs = [
        Document(page_content="Blob storage info", metadata={}),
        Document(page_content="General storage docs", metadata={}),
    ]
    result = generate_response_text(llm_output, docs)

    assert "Azure offers many storage options." in result
    assert "Sources:" not in result