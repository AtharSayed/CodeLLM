# 🎥 Intellitube (Youtube Video Analyzer)

This project  provides a clean Gradio interface to **transcribe,summarize  and interact with YouTube videos**, all **locally and efficiently**, using:

- 🧠 **Faster Whisper** for fast, accurate transcription
- 🤖 **Mistral model via Ollama** for efficient local summarization and Q&A

---

## 🔍 What It Does

Give it any YouTube URL, and it will:

1. Extract and transcribe the video audio using `fast_transcriber` (Faster Whisper)
2. Summarize the transcription using `summarizer` powered by Mistral running locally via Ollama
3. Answers any question related to the video content by using local powered LLM
4. Scrapes Youtube Comment, translate non English comments to English, and performs sentiment analysis
5. Breaks down positive, neutral, and negative audience reactions for deep, real-time insight
6. Display everything  in a beautiful, clean, concise  Gradio web interface
   

---

## ✨ Features

- ⚡ **Fast and local**: Runs without cloud APIs
- 🔐 **Privacy-first**: All processing happens on your machine
- 🧠 **Mistral + Ollama** for efficient LLM inference
- 🎛️ **Gradio UI** with side-by-side transcript and summary
- 🧠 **Multimodal Intelligence** Combines Faster-Whisper for transcription and Deep-Translation for multilingual support for sentiment analysis
- 📦 Modular architecture  for easy integration or expansion

---

## 🚀 Getting Started

### 1. Clone the Main Repo

```bash
git clone https://github.com/AtharSayed/CodeLLM.git
cd CodeLLM/Intellitube
