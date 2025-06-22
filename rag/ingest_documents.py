import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_cache.embedder import get_embedder
from logger.logger import get_logger

# Constants
DOCS_PATH = "data/documents"
VECTOR_STORE_PATH = "data/vector_store"
VECTOR_INDEX_NAME = "azure_docs_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

logger = get_logger(__name__)


def load_all_documents(directory: str):
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
        # DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        # DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
    ]

    docs = []
    for loader in loaders:
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            logger.warning(f"[ingest] Failed to load some files: {e}")
    logger.info(f"[ingest] Loaded {len(docs)} documents.")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def ingest_documents():
    raw_docs = load_all_documents(DOCS_PATH)
    chunks = chunk_documents(raw_docs)

    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vectorstore.save_local(folder_path=VECTOR_STORE_PATH, index_name=VECTOR_INDEX_NAME)

    logger.info(f"[ingest] Saved vector store with {len(chunks)} chunks to {VECTOR_STORE_PATH}/{VECTOR_INDEX_NAME}.faiss")


if __name__ == "__main__":
    ingest_documents()
