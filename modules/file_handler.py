import os
from langchain_community.document_loaders import PyPDFLoader

def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = f"./temp_{uploaded_file.name}"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)
    return documents