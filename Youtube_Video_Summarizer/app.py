import gradio as gr
from fast_transcriber import transcribe_youtube
from summarizer import summarize
from qa_chain import setup_qa
import threading
import traceback

# Global variable to store transcript and QA chain
stored_transcript = None
qa_chain = None

def transcribe_and_summarize(url):
    global stored_transcript, qa_chain
    stored_transcript = None
    qa_chain = None

    transcript = transcribe_youtube(url)

    if transcript.startswith("❌ Error"):
        return transcript, "", "", "❌ Transcript error", ""

    try:
        summary = summarize(transcript)
    except Exception as e:
        summary = f"❌ Summary Error: {str(e)}"

    stored_transcript = transcript  # Save for QA
    return transcript, summary, "", "✅ Transcript and summary ready. You can ask a question now."

def answer_question(question):
    global stored_transcript, qa_chain

    if not question.strip():
        return "❌ Please enter a valid question."
        
    if stored_transcript is None:
        return "❌ No transcript available. Please process a video first."

    # Initialize QA chain if not exists
    if qa_chain is None:
        try:
            qa_chain = setup_qa(stored_transcript)
        except Exception as e:
            return f"❌ Failed to initialize QA system: {str(e)}"

    try:
        # Direct synchronous call with timeout handling
        result = qa_chain.invoke(
            {"query": question},
            config={"max_execution_time": 90}  # 90 second timeout
        )
        return result.get("result", "No answer could be generated.")
        
    except TimeoutError:
        return "❌ The question took too long to answer. Try a simpler question or wait a moment."
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("<h1 style='text-align: center;'>🚀 IntelliTube</h1>", elem_id="title")
        gr.Markdown("<p style='text-align: center;'>Intelligent Video Analysis (Faster Whisper + Mistral via Ollama)</p>")

        url = gr.Textbox(label="YouTube URL", placeholder="🔗 Paste the YouTube video URL here...")

        btn = gr.Button("🎯 Summarize", variant="primary")

        with gr.Row():
            transcript_output = gr.Textbox(label="📝 Transcript", lines=10, interactive=False)
            summary_output = gr.Textbox(label="📌 Summary", lines=10, interactive=False)

        qa_status = gr.Textbox(label="ℹ️ Status", interactive=False)

        question = gr.Textbox(label="❓ Ask a Question", placeholder="e.g. What is the video about?")
        answer_output = gr.Textbox(label="💡 Answer", lines=3, interactive=False)

        btn.click(
            transcribe_and_summarize,
            inputs=url,
            outputs=[transcript_output, summary_output, answer_output, qa_status]
        )

        question.submit(
            answer_question,
            inputs=question,
            outputs=answer_output
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)
