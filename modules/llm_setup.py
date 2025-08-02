from langchain_openai import ChatOpenAI 

def initialize_llm(api_key: str, model_name="gpt-4o"):  # change model_name as needed
    return ChatOpenAI(openai_api_key=api_key, model_name=model_name)