# tests/test_prompt_builder.py

import pytest
from rag.prompt_builder import build_prompt
from langchain_core.documents import Document

def test_prompt_builder_with_context():
    query = "What is Azure Blob Storage?"
    docs = [
        Document(page_content="Azure Blob Storage is used to store unstructured data."),
        Document(page_content="It supports massive scale and high availability."),
    ]
    prompt = build_prompt(query, docs)

    assert "Azure Blob Storage is used to store" in prompt
    assert "What is Azure Blob Storage?" in prompt
    assert "### Documentation:" in prompt
    assert "### User Question:" in prompt
