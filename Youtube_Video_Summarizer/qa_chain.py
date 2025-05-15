from langchain_community.vectorstores import FAISS  # More efficient than Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Initialize LLM
llm = OllamaLLM(model="mistral:7b-instruct-q4_0", temperature=0.1)

# Custom prompt for better QA
prompt_template = """Use the following context to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def setup_qa(transcript: str):
    # Better text splitting with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.create_documents([transcript])
    
    # Efficient embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Better than MiniLM
        model_kwargs={'device': 'cpu'}
    )
    
    # Use FAISS for faster similarity search
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False
    )