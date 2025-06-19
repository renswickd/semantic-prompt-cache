# tests/test_cache_operations.py

import os
import shutil
import pytest
from semantic_cache import operations

TEST_INDEX_DIR = "data/test_cache_store"
TEST_INDEX_NAME = "test_cache"
os.environ["CACHE_THRESHOLD"] = "0.15"

# Sample data
query1 = "best beaches in New Zealand"
response1 = "Piha Beach, Cathedral Cove, Ninety Mile Beach"

query2 = "famous mountains in Italy"
response2 = "Matterhorn, Monte Rosa, Gran Paradiso"


@pytest.fixture(autouse=True)
def clean_test_dir():
    """
    Runs before and after each test to clear the test index directory.
    """
    if os.path.exists(TEST_INDEX_DIR):
        shutil.rmtree(TEST_INDEX_DIR)
    os.makedirs(TEST_INDEX_DIR)
    yield
    if os.path.exists(TEST_INDEX_DIR):
        shutil.rmtree(TEST_INDEX_DIR)


def test_cache_miss_returns_none():
    # Index does not exist → expect miss
    result = operations.get_from_cache(query1)
    assert result is None


def test_cache_insert_and_retrieve():
    # Insert a query-response pair
    operations.set_in_cache(query1, response1)

    # Now retrieval should work
    result = operations.get_from_cache(query1)
    assert result is not None
    assert isinstance(result, str)
    assert response1[:10] in result  # partial match check


def test_cache_insert_multiple_queries():
    operations.set_in_cache(query1, response1)
    operations.set_in_cache(query2, response2)

    result1 = operations.get_from_cache(query1)
    result2 = operations.get_from_cache(query2)

    assert response1[:10] in result1
    assert response2[:10] in result2


def test_cache_clear_empties_all():
    operations.set_in_cache(query1, response1)
    operations.clear_cache()

    result = operations.get_from_cache(query1)
    assert result is None
