import pytest
from rag.llm_client import generate_answer


def test_basic_query_returns_string():
    query = "What is Azure Virtual Network?"
    response = generate_answer(query)
    assert isinstance(response, str)
    assert len(response) > 0


def test_empty_query():
    query = ""
    response = generate_answer(query)
    assert isinstance(response, str)
    assert len(response) > 0  # Should return a helpful fallback message
