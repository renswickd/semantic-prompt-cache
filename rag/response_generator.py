from langchain_core.documents import Document

def generate_response_text(llm_output: str, docs: list[Document]) -> str:
    """
    Optionally formats the LLM response, appending sources.
    """
    response = llm_output.strip()
    
    # Append source metadata
    sources = set()
    for doc in docs:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    if sources:
        source_list = "\n".join([f"- {src}" for src in sources])
        response += f"\n\nSources:\n{source_list}"

    return response


if __name__ == "__main__":
    from langchain_core.documents import Document
    # Example usage
    example_llm_output = "Azure Cognitive Services provides various AI capabilities."
    example_docs = [
        Document(page_content="Azure Cognitive Services includes vision, speech, language, and decision-making APIs.",
                 metadata={"source": "azure_docs.pdf"}),
        Document(page_content="It allows developers to integrate AI into their applications easily.",
                 metadata={"source": "azure_overview.pdf"})
    ]

    response_text = generate_response_text(example_llm_output, example_docs)
    print(response_text)