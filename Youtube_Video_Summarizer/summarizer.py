from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

def summarize(text):
    llm = OllamaLLM(model="mistral:7b-instruct-q4_0")
    docs = [Document(page_content=text[:3000])]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.invoke({"input_documents": docs})["output_text"]