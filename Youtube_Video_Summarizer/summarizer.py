from langchain.llms import Ollama

def summarize_text(text):
    prompt = f"Summarize this video transcript:\n\n{text[:4000]}"
    llm = Ollama(model="mistral")  
    return llm(prompt)
