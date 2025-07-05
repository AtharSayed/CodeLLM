import ollama

def correct_transcript(text, model="mistral"):
    prompt = f"""Correct the grammar, punctuation, and sentence structure of the following transcript.
Keep it natural, but don't change the meaning or tone. Remove filler words if needed.

Transcript:
\"\"\"{text}\"\"\"

Corrected Transcript:"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['message']['content']
