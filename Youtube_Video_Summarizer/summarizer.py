from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def summarize_text(text):
    llm = Ollama(model="mistral:7b-instruct-q4_0")  # quantized version
    chunks = [Document(page_content=chunk) for chunk in text.split("\n\n") if chunk.strip()]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(chunks)
