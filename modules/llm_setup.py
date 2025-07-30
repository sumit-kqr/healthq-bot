from langchain_groq import ChatGroq

def initialize_llm(api_key: str, model_name="Gemma2-9b-it"):
    return ChatGroq(groq_api_key=api_key, model_name=model_name)
