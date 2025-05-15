import gradio as gr
from fast_transcriber import transcribe_youtube
from summarizer import summarize  # Import the summarize function

def transcribe_and_summarize(url):
    transcript = transcribe_youtube(url)
    
    if transcript.startswith("❌ Error"):
        return transcript, ""
    
    try:
        summary = summarize(transcript)
    except Exception as e:
        summary = f"❌ Summary Error: {str(e)}"
    
    return transcript, summary

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("<h1 style='text-align: center;'>🚀 IntelliTube</h1>", elem_id="title")
        gr.Markdown("<p style='text-align: center;'>Intelligent Video Analysis (Faster Whisper + Mistral via Ollama)</p>")
        
        url = gr.Textbox(
            label="YouTube URL", 
            placeholder="🔗 Paste the YouTube video URL here..."
        )
        
        btn = gr.Button("🎯 Summarize", variant="primary")
        
        with gr.Row():
            transcript_output = gr.Textbox(label="📝 Transcript", lines=10, interactive=False)
            summary_output = gr.Textbox(label="📌 Summary", lines=10, interactive=False)
        
        btn.click(
            transcribe_and_summarize,
            inputs=url,
            outputs=[transcript_output, summary_output]
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)
