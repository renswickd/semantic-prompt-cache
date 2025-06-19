import pytest
import shutil
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from semantic_cache.index_manager import (
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
    reset_faiss_index,
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_NAME
)


@pytest.fixture(scope="module")
def test_data():
    return {
        "texts": [
            "What is the capital of France?",
            "How does semantic caching work?"
        ],
        "metadatas": [
            {"source": "doc1"},
            {"source": "doc2"}
        ]
    }


@pytest.fixture(scope="module")
def setup_test_index(test_data):
    # Ensure clean state
    reset_faiss_index(DEFAULT_INDEX_DIR)

    # Create index
    index = create_faiss_index(
        texts=test_data["texts"],
        metadatas=test_data["metadatas"]
    )
    yield index

    # Teardown
    reset_faiss_index(DEFAULT_INDEX_DIR)


def test_create_index(setup_test_index, test_data):
    index = setup_test_index
    assert index is not None
    assert index.index is not None
    assert index.docstore is not None
    assert len(index.docstore._dict) == len(test_data["texts"])


def test_load_index(test_data):
    index = load_faiss_index()
    assert index is not None
    assert len(index.docstore._dict) == len(test_data["texts"])


def test_save_and_reload_index(test_data):
    index = create_faiss_index(test_data["texts"], test_data["metadatas"])
    save_faiss_index(index, DEFAULT_INDEX_DIR, DEFAULT_INDEX_NAME)

    reloaded = load_faiss_index()
    assert reloaded is not None
    assert reloaded.index.ntotal == len(test_data["texts"])
    assert list(reloaded.docstore._dict.values())[0].metadata["source"] in ["doc1", "doc2"]


def test_reset_index():
    reset_faiss_index(DEFAULT_INDEX_DIR)
    index_path = Path(DEFAULT_INDEX_DIR)
    files = list(index_path.glob("*"))
    assert len(files) == 0
