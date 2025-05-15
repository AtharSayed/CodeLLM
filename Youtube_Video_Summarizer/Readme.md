# 🎥 Youtube Video Summarizer

This project is a submodule of the [CodeLLM](https://github.com/AtharSayed/CodeLLM) repo. It provides a clean Gradio interface to **transcribe and summarize YouTube videos**, all **locally and efficiently**, using:

- 🧠 **Faster Whisper** for fast, accurate transcription
- 🤖 **Mistral model via Ollama** for efficient local summarization

---

## 🔍 What It Does

Give it any YouTube URL, and it will:

1. Extract and transcribe the video audio using `fast_transcriber` (Faster Whisper)
2. Summarize the transcription using `summarizer` powered by Mistral running locally via Ollama
3. Display both in a beautiful Gradio web interface

---

## ✨ Features

- ⚡ **Fast and local**: Runs without cloud APIs
- 🔐 **Privacy-first**: All processing happens on your machine
- 🧠 **Mistral + Ollama** for efficient LLM inference
- 🎛️ **Gradio UI** with side-by-side transcript and summary
- 📦 Modular structure for easy integration or expansion

---

## 🚀 Getting Started

### 1. Clone the Main Repo

```bash
git clone https://github.com/AtharSayed/CodeLLM.git
cd CodeLLM/Youtube_Video_Summarizer
