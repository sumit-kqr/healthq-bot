# import os
# from langchain_community.document_loaders import PyPDFLoader

# def load_documents(uploaded_files):
#     documents = []
#     for uploaded_file in uploaded_files:
#         temp_pdf = f"./temp_{uploaded_file.name}"
#         with open(temp_pdf, "wb") as f:
#             f.write(uploaded_file.getvalue())
#         loader = PyPDFLoader(temp_pdf)
#         docs = loader.load()
#         documents.extend(docs)
#     return documents
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from typing import Iterable, Union

try:
    # Streamlit is optional; used only to detect UploadedFile type
    import streamlit as st  # type: ignore
    UploadedFileType = st.runtime.uploaded_file_manager.UploadedFile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best-effort optional import
    UploadedFileType = object  # Fallback to avoid type errors if Streamlit isn't present


def load_documents(files_or_paths: Iterable[Union[str, "UploadedFileType"]]):
    documents = []
    for item in files_or_paths:
        # Handle Streamlit UploadedFile
        if hasattr(item, "getvalue") and hasattr(item, "name"):
            file_bytes = item.getvalue()
            original_name = getattr(item, "name", "upload.pdf")
            suffix = ".pdf" if str(original_name).lower().endswith(".pdf") else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
        # Handle string file path
        elif isinstance(item, str):
            loader = PyPDFLoader(item)
        else:
            raise ValueError("Unsupported file input type; expected file path or Streamlit UploadedFile")

        docs = loader.load()
        documents.extend(docs)

    return documents
