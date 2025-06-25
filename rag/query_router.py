from semantic_cache.operations import get_from_cache, set_in_cache
from rag.retriever import retrieve_relevant_docs
from rag.prompt_builder import build_prompt
from rag.llm_client import generate_answer
from rag.response_generator import generate_response_text
from logger.logger import get_logger

logger = get_logger(__name__)

def query_router(user_query: str) -> str:
    """
    End-to-end pipeline:
    1. Check semantic cache
    2. If miss, do RAG: retrieval + prompt + LLM
    3. Store in cache
    4. Return final result
    """
    logger.info("[router] Received query")

    # Step 1: Semantic Cache
    cached_response = get_from_cache(user_query)
    if cached_response:
        logger.info("[router] Returning cached response")
        return cached_response

    # Step 2: RAG Retrieval
    docs = retrieve_relevant_docs(user_query)
    logger.info(f"[router] Retrieved {len(docs)} context docs")

    # Step 3: Prompt Construction
    prompt = build_prompt(user_query, docs)

    # Step 4: LLM Call
    llm_output = generate_answer(prompt)

    # Step 5: Response Formatting
    final_response = generate_response_text(llm_output, docs)

    # Step 6: Cache this response
    set_in_cache(user_query, final_response)

    return final_response

if __name__ == "__main__":
    # Example usage
    from semantic_cache.operations import clear_cache

    # Example query to test the router
    example_query = "What are the key features of Azure Cognitive Services?"
    response = query_router(example_query)
    print(f"Response for query '{example_query}': {response}")

    clear_cache() # Clear cache after testing