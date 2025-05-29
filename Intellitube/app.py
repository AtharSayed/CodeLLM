import gradio as gr
from fast_transcriber import transcribe_youtube
from transcorrection import correct_transcript
from summarizer import summarize
from qa_chain import setup_qa
from ytsenti import fetch_comments_scrape, analyze_sentiment, analyze_intent
import threading
import traceback
import json
import os
import re
from datetime import datetime

# Global variables
stored_transcript = None
qa_chain = None
question_history = []

# ---------------------------
# ✅ YouTube URL Validation
# ---------------------------
def is_valid_youtube_url(url):
    if not url:
        return False
    youtube_regex = re.compile(
        r"(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+"
    )
    return youtube_regex.match(url) is not None

# ---------------------------
# 🔁 Transcription & Summarization
# ---------------------------
def transcribe_and_summarize(url, progress=gr.Progress()):
    global stored_transcript, qa_chain
    stored_transcript = None
    qa_chain = None

    if not is_valid_youtube_url(url):
        return "", "", "", "❌ Invalid or missing YouTube URL.", "", None

    progress(0, desc="Starting transcription...")

    try:
        progress(0.3, desc="Transcribing YouTube video...")
        raw_transcript = transcribe_youtube(url)

        if raw_transcript.startswith("❌ Error"):
            return raw_transcript, "", "", "❌ Transcript error", "", None

        progress(0.5, desc="Correcting transcript...")
        
        # Without transcript correction layer
        #transcript = raw_transcript

        # With transcript correction layer 
        transcript = correct_transcript(raw_transcript)

        progress(0.7, desc="Generating summary...")
        try:
            summary = summarize(transcript)
        except Exception as e:
            summary = f"❌ Summary Error: {str(e)}"

        stored_transcript = transcript
        progress(1.0, desc="Completed!")
        return transcript, summary, "", "✅ Transcript and summary ready. You can ask a question now.", "", None

    except Exception as e:
        return "", "", "", f"❌ Error: {str(e)}", "", None

# ---------------------------
# ❓ Question Answering
# ---------------------------
def answer_question(question):
    global stored_transcript, qa_chain, question_history

    if not question.strip():
        return "❌ Please enter a valid question.", format_question_history()
    
    if stored_transcript is None:
        return "❌ No transcript available. Please process a video first.", format_question_history()

    if qa_chain is None:
        try:
            qa_chain = setup_qa(stored_transcript)
        except Exception as e:
            return f"❌ Failed to initialize QA system: {str(e)}", format_question_history()

    try:
        result = qa_chain.invoke(
            {"query": question},
            config={"max_execution_time": 90}
        )
        answer = result.get("result", "No answer could be generated.")
        question_history.append({"question": question, "answer": answer})
        return answer, format_question_history()
        
    except TimeoutError:
        return "❌ The question took too long to answer. Try a simpler question.", format_question_history()
    except Exception as e:
        return f"❌ Error answering question: {str(e)}", format_question_history()

# ---------------------------
# 📝 Question History Formatter
# ---------------------------
def format_question_history():
    if not question_history:
        return "No questions asked yet."
    history_text = ""
    for i, item in enumerate(question_history, 1):
        history_text += f"**Q{i}:** {item['question']}\n**A{i}:** {item['answer']}\n\n"
    return history_text

# ---------------------------
# 💬 Comment Sentiment Analysis
# ---------------------------
def analyze_comments_sentiment(url, progress=gr.Progress()):
    if not is_valid_youtube_url(url):
        return "❌ Invalid or missing YouTube URL.", None

    progress(0, desc="Fetching comments...")
    comments = fetch_comments_scrape(url, max_comments=50)
    if not comments:
        return "❌ Failed to fetch comments or no comments found.", None

    progress(0.5, desc="Analyzing sentiment and intent...")
    sentiment_summary, _ = analyze_sentiment(comments)
    intent_summary, _ = analyze_intent(comments)

    sentiment_text = (
        f"🟢 POSITIVE: {sentiment_summary['POSITIVE']}\n"
        f"🟡 NEUTRAL: {sentiment_summary['NEUTRAL']}\n"
        f"🔴 NEGATIVE: {sentiment_summary['NEGATIVE']}\n\n"
    )

    if sentiment_summary["POSITIVE"] > sentiment_summary["NEGATIVE"]:
        sentiment_text += "✅ Overall Sentiment: Mostly Positive\n\n"
    elif sentiment_summary["NEGATIVE"] > sentiment_summary["POSITIVE"]:
        sentiment_text += "⚠️ Overall Sentiment: Mostly Negative\n\n"
    else:
        sentiment_text += "📊 Overall Sentiment: Mixed or Neutral\n\n"

    sentiment_text += "🎯 Intent Summary:\n"
    for intent, count in intent_summary.items():
        sentiment_text += f"🔹 {intent.upper()}: {count}\n"

    progress(1.0, desc="Completed!")
    return sentiment_text, None

# ---------------------------
# 💾 Save Output to File
# ---------------------------
def save_outputs(transcript, summary, sentiment):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "intellitube_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "transcript": transcript,
        "summary": summary,
        "sentiment": sentiment,
        "question_history": question_history
    }
    
    file_path = os.path.join(output_dir, f"output_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return f"✅ Saved to {file_path}"

# ---------------------------
# 🧹 Clear Everything
# ---------------------------
def clear_all():
    global stored_transcript, qa_chain, question_history
    stored_transcript = None
    qa_chain = None
    question_history = []
    return "", "", "", "", "", "No questions asked yet.", None

# ---------------------------
# 🚀 Gradio UI
# ---------------------------
with gr.Blocks(theme=gr.themes.Soft(), css=""" 
    #title { 
        font-family: 'Inter', sans-serif; 
        font-size: 2.5em; 
        text-align: center; 
        margin-bottom: 10px; 
        font-weight: 700; 
    }
    .gr-button { 
        border-radius: 8px; 
        font-family: 'Inter', sans-serif; 
        font-size: 1.1em; 
        padding: 10px; 
    }
    .gr-textbox { 
        border-radius: 8px; 
        font-family: 'Inter', sans-serif; 
        font-size: 1.1em; 
        line-height: 1.5; 
    }
    .gr-markdown { 
        font-family: 'Inter', sans-serif; 
        font-size: 1.1em; 
        line-height: 1.6; 
    }
    .output-text { 
        font-size: 1.2em; 
        line-height: 1.6; 
        background-color: #f9f9f9; 
        padding: 15px; 
        border-radius: 8px; 
    }
    .status-box { 
        background-color: #f0f0f0; 
        padding: 10px; 
        border-radius: 8px; 
        font-family: 'Inter', sans-serif; 
        font-size: 1.1em; 
    }
    .history-text { 
        font-family: 'Inter', sans-serif; 
        font-size: 1.1em; 
        line-height: 1.6; 
        background-color: #f9f9f9; 
        padding: 15px; 
        border-radius: 8px; 
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
""") as demo:
    gr.Markdown("<h1 id='title'>🚀 IntelliTube</h1>")
    gr.Markdown("<p style='text-align: center;'>Intelligent Video Analysis with Faster Whisper + Mistral</p>")

    with gr.Column():
        url = gr.Textbox(label="YouTube URL", placeholder="🔗 Paste the YouTube video URL here...", lines=1)
        
        with gr.Row():
            process_btn = gr.Button("🎯 Process Video", variant="primary")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        status = gr.Textbox(label="ℹ️ Status", interactive=False, elem_classes="status-box")

    with gr.Tabs():
        with gr.TabItem("📝 Transcript"):
            transcript_output = gr.Textbox(label="Transcript", lines=10, interactive=False, elem_classes="output-text")
            save_transcript_btn = gr.Button("💾 Save Outputs")
        
        with gr.TabItem("📌 Summary"):
            summary_output = gr.Textbox(label="Summary", lines=10, interactive=False, elem_classes="output-text")
            save_summary_btn = gr.Button("💾 Save Outputs")
        
        with gr.TabItem("❓ Q&A"):
            question = gr.Textbox(label="Ask a Question", placeholder="e.g., What is the video about?")
            answer_output = gr.Textbox(label="Answer", lines=5, interactive=False, elem_classes="output-text")
            question_history_output = gr.Textbox(label="Question History", lines=10, interactive=False, elem_classes="history-text")
            with gr.Row():
                ask_btn = gr.Button("✅ Submit Question", variant="primary")
                copy_answer_btn = gr.Button("📋 Copy Answer")
        
        with gr.TabItem("💬 Sentiment"):
            sentiment_output = gr.Textbox(label="Comment Sentiment", lines=10, interactive=False, elem_classes="output-text")
            sentiment_btn = gr.Button("🧠 Analyze Comments")
            save_sentiment_btn = gr.Button("💾 Save Outputs")

    # Event bindings
    process_btn.click(
        transcribe_and_summarize,
        inputs=url,
        outputs=[transcript_output, summary_output, answer_output, status, sentiment_output, question_history_output]
    )

    ask_btn.click(
        answer_question,
        inputs=question,
        outputs=[answer_output, question_history_output]
    )

    question.submit(
        answer_question,
        inputs=question,
        outputs=[answer_output, question_history_output]
    )

    sentiment_btn.click(
        analyze_comments_sentiment,
        inputs=url,
        outputs=[sentiment_output, status]
    )

    clear_btn.click(
        clear_all,
        inputs=None,
        outputs=[url, transcript_output, summary_output, answer_output, sentiment_output, question_history_output, status]
    )

    save_transcript_btn.click(
        save_outputs,
        inputs=[transcript_output, summary_output, sentiment_output],
        outputs=status
    )

    save_summary_btn.click(
        save_outputs,
        inputs=[transcript_output, summary_output, sentiment_output],
        outputs=status
    )

    save_sentiment_btn.click(
        save_outputs,
        inputs=[transcript_output, summary_output, sentiment_output],
        outputs=status
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
