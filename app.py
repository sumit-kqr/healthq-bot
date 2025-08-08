import streamlit as st
import os
from dotenv import load_dotenv

from modules.llm_setup import initialize_llm
from modules.file_handler import load_documents
from modules.vector_store import build_vectorstore
from modules.retriever_chain import build_conversational_rag_chain
from modules.session_handler import get_session_history

# Load environment variables
load_dotenv()
# Optional: set HF token only if provided
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

st.title("HealthQ ChatBot")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    llm = initialize_llm(api_key)

    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = load_documents(uploaded_files)
        vectorstore = build_vectorstore(documents)
        retriever = vectorstore.as_retriever()

        conversational_chain = build_conversational_rag_chain(
            llm, retriever, lambda s: get_session_history(st.session_state, s)
        )

        user_input = st.text_input("Your question")
        if user_input:
            session_history = get_session_history(st.session_state, session_id)
            try:
                response = conversational_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.success(response["answer"])
               # st.write("Chat history:", session_history.messages)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("something went wrong , server error")
