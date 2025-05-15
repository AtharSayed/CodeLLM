# Question-Answering based on the Video Transcript 

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

def setup_qa(transcript):
    chunks = [Document(page_content=transcript[i:i+500]) 
              for i in range(0, len(transcript), 500)]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_documents(chunks, embeddings)
    
    return RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="mistral:7b-instruct-q4_0"),
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )