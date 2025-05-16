# 🎥 Intellitube (Youtube Video Analyzer)

This project is a submodule of the [CodeLLM](https://github.com/AtharSayed/CodeLLM) repo. It provides a clean Gradio interface to **transcribe,summarize  and interact with YouTube videos**, all **locally and efficiently**, using:

- 🧠 **Faster Whisper** for fast, accurate transcription
- 🤖 **Mistral model via Ollama** for efficient local summarization and Q&A

---

## 🔍 What It Does

Give it any YouTube URL, and it will:

1. Extract and transcribe the video audio using `fast_transcriber` (Faster Whisper)
2. Summarize the transcription using `summarizer` powered by Mistral running locally via Ollama
3. Answers any question related to the video content 
4. Display both in a beautiful, clean Gradio web interface
   

---

## ✨ Features

- ⚡ **Fast and local**: Runs without cloud APIs
- 🔐 **Privacy-first**: All processing happens on your machine
- 🧠 **Mistral + Ollama** for efficient LLM inference
- 🎛️ **Gradio UI** with side-by-side transcript and summary
- 📦 Modular architecture  for easy integration or expansion

---

## 🚀 Getting Started

### 1. Clone the Main Repo

```bash
git clone https://github.com/AtharSayed/CodeLLM.git
cd CodeLLM/Intellitube
