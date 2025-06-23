# RAG + Semantic Cache System

This project is designed to enhance a Retrieval-Augmented Generation (RAG) pipeline with a custom-built Semantic Cache system. The primary goal is to reduce redundant LLM (Large Language Model) calls, improve system responsiveness, and optimize cost for real-time and large-scale AI applications.

## 🚀 Purpose

In traditional RAG pipelines, every user query is processed through document retrieval and LLM generation—even if a semantically similar query was already answered. This approach increases latency and inflates API usage costs.

This system introduces a semantic caching layer that intercepts incoming queries and compares them—based on meaning, not just keywords—against previously answered queries. If a sufficiently similar query is found, the cached response is reused, bypassing the need for another LLM call.

## 🔧 Use Cases

- **Chatbots with memory efficiency**  
  Minimize repeated LLM calls for frequently asked or rephrased questions.

- **Enterprise knowledge assistants**  
  Provide consistent and faster answers to similar user queries across departments.

- **High-throughput RAG pipelines**  
  Scale to thousands of queries per day while maintaining performance and reducing cost.

- **Latency-sensitive applications**  
  Reduce end-user wait time by short-circuiting the full RAG flow when a cached response is available.


# Semantic Cache for LLM-Enhanced RAG

A modular, non-OOP semantic caching system built to reduce LLM calls and latency in Retrieval-Augmented Generation (RAG) pipelines.

## 🔧 Features

- ✅ Embeds user queries using `bge-small-en-v1.5`
- ✅ Stores query-response pairs with FAISS index
- ✅ Retrieves cached results based on semantic similarity
- ✅ Configurable similarity threshold
- ✅ Supports metadata (timestamps, hits) and leaderboard extensions
- ✅ Fully functional with Mistral (via Groq) or any OpenRouter-compatible LLM
- ✅ Enterprise knowledge assistants (e.g. Azure Docs)
- ✅ High-throughput RAG pipelines
- ✅ Latency-sensitive LLM apps

# 🧱 Architecture Overview

```text
            ┌──────────────────────────────┐
            │        User Query Input       │
            └──────────────────────────────┘
                         │
                         ▼
     ┌───────────────────────────────────────┐
     │ 1. Check Semantic Cache (FAISS)       │
     └───────────────────────────────────────┘
         │ Yes (high match)   │ No (miss)
         ▼                    ▼
  Reuse Cached LLM     ┌─────────────────────┐
      Response         │ 2. Retrieve Context │
                       └─────────────────────┘
                               │
                               ▼
         ┌────────────────────────────────┐
         │ 3. Build Prompt + Inject Docs  │
         └────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────┐
        │ 4. Generate Response (Mistral LLM) │
        └────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────┐
        │ 5. Postprocess + Store in Cache    │
        └────────────────────────────────────┘
```

## 📁 Key Modules

| Module | Purpose |
|--------|---------|
| `semantic_cache/embedder.py` | Loads BGE model and returns query embeddings |
| `semantic_cache/index_manager.py` | Manages FAISS index creation, loading, saving |
| `semantic_cache/operations.py` | Handles get/set/clear cache operations |
| `rag/retriever.py` |	Top-k document retrieval from Azure knowledge base |
| `rag/prompt_builder.py` |	Combines retrieved chunks + user question into LLM prompt |
| `rag/llm_client.py` |	Calls Mistral via Groq using LangChain |
| `rag/ingest_docs.py` |	Preprocesses and uploads local docs into FAISS vectorstore |
| `tests/` | Unit tests for all core functionality |

## 🚀 Usage (Example)

```python
from semantic_cache.operations import get_from_cache, set_in_cache

query = "top places to visit in France"
cached = get_from_cache(query)

if cached:
    print("✅ Cache Hit:", cached)
else:
    response = "Paris, Lyon, Nice..."  
    set_in_cache(query, response)
```

## Run Tests
```python
pytest tests/
```


---
## 📌 Next Steps
🔁 Add leaderboard and TTL/size-based cache trimming

📚 Ingest Azure PDF documentation automatically

🌐 Wrap with FastAPI for API serving

☁️ Upgrade from FAISS → Qdrant/Chroma

🤖 Migrate from Groq to AI Foundry (multi-LLM orchestration)

