import os
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from logger.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Groq Mistral LLM Initialization
llm = ChatGroq(
    model_name="llama3-70b-8192", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2 
)

# Default prompt structure
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for answering questions about Azure."),
    ("user", "{query}")
])

# Create LangChain Runnable pipeline
rag_chain: Runnable = prompt | llm


def generate_answer(query: str) -> str:
    """
    Calls Groq (Mistral) LLM with the given query and returns response text.
    """
    try:
        result = rag_chain.invoke({"query": query})
        logger.info("[llm_client] Received response from LLM.")
        return result.content
    except Exception as e:
        logger.error(f"[llm_client] LLM call failed: {e}")
        return "Sorry, an error occurred while generating the response."

if __name__ == "__main__":
    # Example usage
    example_query = "What are the key features of Azure Cognitive Services?"
    response = generate_answer(example_query)
    print(f"Response: {response}")

























#####################################
# ======= LangChain Groq Client =======#
#####################################
# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# import json
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Initialize Groq LLM
# llm = ChatGroq(
#     model_name="llama-3.3-70b-versatile",
#     temperature=0.7,
#     api_key=os.getenv("GROQ_API_KEY"),
# )

# # Define the expected JSON structure
# parser = JsonOutputParser(pydantic_object={
#     "type": "object",
#     "properties": {
#         "name": {"type": "string"},
#         "price": {"type": "number"},
#         "features": {
#             "type": "array",
#             "items": {"type": "string"}
#         }
#     }
# })

# # Create a simple prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """Extract product details into JSON with this structure:
#         {{
#             "name": "product name here",
#             "price": number_here_without_currency_symbol,
#             "features": ["feature1", "feature2", "feature3"]
#         }}"""),
#     ("user", "{input}")
# ])

# # Create the chain that guarantees JSON output
# chain = prompt | llm | parser

# def parse_product(description: str) -> dict:
#     result = chain.invoke({"input": description})
#     print(json.dumps(result, indent=2))


# if __name__ == "__main__":      
#     # Example usage
#     description = """The Kees Van Der Westen Speedster is a high-end, single-group espresso machine known for its precision, performance, 
#     and industrial design. Handcrafted in the Netherlands, it features dual boilers for brewing and steaming, PID temperature control for 
#     consistency, and a unique pre-infusion system to enhance flavor extraction. Designed for enthusiasts and professionals, it offers 
#     customizable aesthetics, exceptional thermal stability, and intuitive operation via a lever system. The pricing is approximatelyt $14,499 
#     depending on the retailer and customization options."""

#     parse_product(description)
