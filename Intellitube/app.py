import gradio as gr
from fast_transcriber import transcribe_youtube
from summarizer import summarize
from qa_chain import setup_qa
from ytsenti import fetch_comments_scrape, analyze_sentiment,analyze_intent  # ✅ Import sentiment functions

import threading
import traceback

# Global variables
stored_transcript = None
qa_chain = None

# Transcription & Summarization
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

    stored_transcript = transcript
    return transcript, summary, "", "✅ Transcript and summary ready. You can ask a question now.", ""

# Question Answering
def answer_question(question):
    global stored_transcript, qa_chain

    if not question.strip():
        return "❌ Please enter a valid question."
        
    if stored_transcript is None:
        return "❌ No transcript available. Please process a video first."

    if qa_chain is None:
        try:
            qa_chain = setup_qa(stored_transcript)
        except Exception as e:
            return f"❌ Failed to initialize QA system: {str(e)}"

    try:
        result = qa_chain.invoke(
            {"query": question},
            config={"max_execution_time": 90}
        )
        return result.get("result", "No answer could be generated.")
        
    except TimeoutError:
        return "❌ The question took too long to answer. Try a simpler question or wait a moment."
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"

# Comment Sentiment Analysis
def analyze_comments_sentiment(url):
    comments = fetch_comments_scrape(url, max_comments=50)
    if not comments:
        return "❌ Failed to fetch comments or no comments found."

    sentiment_summary, _ = analyze_sentiment(comments)
    intent_summary, _ = analyze_intent(comments)

    # Build sentiment summary
    sentiment_text = (
        f"🟢 POSITIVE: {sentiment_summary['POSITIVE']}\n"
        f"🟡 NEUTRAL : {sentiment_summary['NEUTRAL']}\n"
        f"🔴 NEGATIVE: {sentiment_summary['NEGATIVE']}\n\n"
    )

    if sentiment_summary["POSITIVE"] > sentiment_summary["NEGATIVE"]:
        sentiment_text += "✅ Overall Sentiment: Mostly Positive\n\n"
    elif sentiment_summary["NEGATIVE"] > sentiment_summary["POSITIVE"]:
        sentiment_text += "⚠️ Overall Sentiment: Mostly Negative\n\n"
    else:
        sentiment_text += "📊 Overall Sentiment: Mixed or Neutral\n\n"

    # Build intent summary
    sentiment_text += "🎯 Intent Summary:\n"
    for intent, count in intent_summary.items():
        sentiment_text += f"🔹 {intent.upper()}: {count}\n"

    return sentiment_text

# Gradio UI
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

        sentiment_button = gr.Button("🧠 Analyze YouTube Comments for sentiment")
        sentiment_output = gr.Textbox(label="💬 Comment Sentiment", lines=10, interactive=False)

        # Event bindings
        btn.click(
            transcribe_and_summarize,
            inputs=url,
            outputs=[transcript_output, summary_output, answer_output, qa_status, sentiment_output]
        )

        question.submit(
            answer_question,
            inputs=question,
            outputs=answer_output
        )

        sentiment_button.click(
            analyze_comments_sentiment,
            inputs=url,
            outputs=sentiment_output
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)
