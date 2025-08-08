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

st.set_page_config(page_title="HealthQ ChatBot", page_icon="🩺", layout="centered")
st.title("HealthQ ChatBot")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY. Please set the environment variable and reload.")
    st.stop()

# Initialize LLM once
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm(api_key)

# Initialize storage for chat and artifacts
if "store" not in st.session_state:
    st.session_state.store = {}
if "files_signature" not in st.session_state:
    st.session_state.files_signature = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        process_clicked = st.button("Build knowledge base", use_container_width=True)
    with col_b:
        if st.button("Reset session", use_container_width=True):
            for key in [
                "files_signature",
                "retriever",
                "rag_chain",
            ]:
                st.session_state.pop(key, None)
            st.toast("Session reset.")

def _signature_for_files(files):
    if not files:
        return None
    try:
        # Stable lightweight signature: (name, size) for each file
        return tuple((getattr(f, "name", str(i)), getattr(f, "size", 0)) for i, f in enumerate(files))
    except Exception:
        return None

if process_clicked and uploaded_files:
    with st.spinner("Processing documents and building vector store..."):
        sig = _signature_for_files(uploaded_files)
        if sig != st.session_state.files_signature:
            documents = load_documents(uploaded_files)
            vectorstore = build_vectorstore(documents)
            st.session_state.retriever = vectorstore.as_retriever()
            st.session_state.rag_chain = build_conversational_rag_chain(
                st.session_state.llm,
                st.session_state.retriever,
                lambda s: get_session_history(st.session_state, s),
            )
            st.session_state.files_signature = sig
            st.success("Knowledge base ready.")
        else:
            st.info("Same files detected. Reusing existing knowledge base.")

session_id = st.text_input("Session ID", value="default_session")
user_input = st.text_input("Your question")

if user_input:
    if st.session_state.rag_chain is None:
        st.warning("Please upload and process documents first.")
    else:
        try:
            response = st.session_state.rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.success(response["answer"])
        except Exception as e:
            st.error(f"Error: {e}")
