import gradio as gr
from fast_transcriber import transcribe_youtube
from summarizer import summarize  # Import the summarize function

def transcribe_and_summarize(url):
    transcript = transcribe_youtube(url)
    
    # Check if transcription was successful before summarizing
    if transcript.startswith("❌ Error"):
        return transcript, ""
    
    try:
        summary = summarize(transcript)
    except Exception as e:
        summary = f"❌ Summary Error: {str(e)}"
    
    return transcript, summary

with gr.Blocks() as demo:
    gr.Markdown("## 🚀IntelliTube (Faster Whisper + Mistral via Ollama)")
    
    url = gr.Textbox(label="YouTube URL", placeholder="Paste URL here...")
    transcript_output = gr.Textbox(label="Transcript", lines=10)
    summary_output = gr.Textbox(label="Summary", lines=7)
    
    btn = gr.Button("Summarize", variant="primary")
    
    btn.click(
        transcribe_and_summarize,
        inputs=url,
        outputs=[transcript_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
