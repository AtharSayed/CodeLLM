import gradio as gr
from transcriber import download_and_transcribe_youtube
from summarizer import summarize_text
from qa_chain import setup_qa_chain

qa_chain = [None]

def process_video(url):
    transcript = download_and_transcribe_youtube(url)
    summary = summarize_text(transcript)
    qa_chain[0] = setup_qa_chain(transcript)
    return summary

def answer_question(question):
    if qa_chain[0]:
        return qa_chain[0].run(question)
    return "Please process a video first."

with gr.Blocks() as demo:
    gr.Markdown("# 🎥 YouTube Video Summarizer & Q&A")
    url_input = gr.Textbox(label="YouTube URL")
    submit_button = gr.Button("Summarize")
    summary_output = gr.Textbox(label="Summary")

    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")

    submit_button.click(process_video, inputs=url_input, outputs=summary_output)
    question_input.submit(answer_question, inputs=question_input, outputs=answer_output)

demo.launch()
