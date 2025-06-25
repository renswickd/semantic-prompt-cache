from rag.query_router import query_router
from semantic_cache.operations import clear_cache
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # Clear cache at startup
    clear_cache()
    print("Welcome to the RAG Query System!")
    print("Type 'exit' or 'quit' to stop the conversation.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        answer = query_router(query)
        print("\nAI:", answer)