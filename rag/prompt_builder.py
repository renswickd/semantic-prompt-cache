from langchain_core.documents import Document

def build_prompt(query: str, context_docs: list[Document]) -> str:
    """
    Builds the final prompt for the LLM by combining context + user question.
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""You are an AI assistant helping users with Microsoft Azure.

    Use the following documentation excerpts to answer the user's question.

    ### Documentation:
    {context}

    ### User Question:
    {query}

    ### Answer:
    """
    return prompt

if __name__ == "__main__":
    # Example usage
    from rag.retriever import retrieve_relevant_docs

    example_docs = retrieve_relevant_docs("What are the key features of Azure Cognitive Services?")
    print(f"Retrieved {len(example_docs)} documents for the example query.")
    print("Example documents:")
    for doc in example_docs:
        print(f"- {doc.metadata.get('title', 'No title')}: {doc.page_content[:100]}...")
        
    if not example_docs:
        print("No relevant documents found.")
        exit(1)
    
    example_query = "What are the key features of Azure Cognitive Services?"

    final_prompt = build_prompt(example_query, example_docs)
    print(final_prompt)