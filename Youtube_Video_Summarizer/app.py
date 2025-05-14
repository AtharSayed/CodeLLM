import gradio as gr
from transcriber import download_and_transcribe_youtube
from summarizer import summarize_text
from qa_chain import setup_qa_chain

# Simple state management
current_qa_chain = None

def process_video(url):
    global current_qa_chain
    try:
        print("📥 Downloading audio...")
        transcript = download_and_transcribe_youtube(url)
        
        if not transcript or transcript.startswith("Failed"):
            return "Error: Could not process video"
            
        print("🧠 Generating summary...")
        summary = summarize_text(transcript[:3000])  # Use first 3000 chars for speed
        
        print("🔧 Setting up Q&A...")
        current_qa_chain = setup_qa_chain(transcript)
        
        return summary
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error processing video: {str(e)}"

def answer_question(question):
    if not current_qa_chain:
        return "Please process a video first"
    try:
        return current_qa_chain.run(question)
    except Exception as e:
        return f"Error answering: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## YouTube Video Processor")
    
    with gr.Tab("Process Video"):
        url_input = gr.Textbox(label="YouTube URL")
        process_btn = gr.Button("Process")
        summary_output = gr.Textbox(label="Summary", lines=5)
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", lines=3)
    
    process_btn.click(
        process_video,
        inputs=url_input,
        outputs=summary_output
    )
    
    ask_btn.click(
        answer_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)