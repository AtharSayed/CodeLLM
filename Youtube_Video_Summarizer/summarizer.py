from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def summarize_text(text):
    from langchain_community.llms import Ollama
    llm = Ollama(model="mistral:7b-instruct-q4_0")
    chunks = [Document(page_content=text[:3000])]  # Limit input size
    chain = load_summarize_chain(llm, chain_type="stuff")  # Simple "stuff" method
    return chain.run(chunks)