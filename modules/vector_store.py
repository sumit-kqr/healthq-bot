from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="./chroma_db" )
    return vectorstore
