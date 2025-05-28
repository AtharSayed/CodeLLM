# 🎥 Intellitube (Youtube Video Analyzer)

This project features a **sleek and modern web interface** for **transcribing, summarizing, and analyzing YouTube videos**, all **locally and efficiently**, using:

- 🧠 **Faster Whisper** for fast, accurate transcription
- 🤖 **Mistral 7B quantized model via Ollama** for efficient local summarization and Q&A

---

## 🔍 What It Does

Give it any YouTube URL, and it will:

1.  Extract and transcribe the video audio using `fast_transcriber` (Faster Whisper).
2.  Summarize the transcription using `summarizer` powered by Mistral running locally via Ollama.
3.  Answers any question related to the video content by using local powered LLM.
4.  Scrapes Youtube Comment, translate non English comments to English, and performs sentiment analysis.
5.  Breaks down positive, neutral, and negative audience reactions for deep, real-time insight.
6.  Provides a secure and intuitive **login and sign-up experience** for user management.
7.  Displays all analysis results within a beautiful, clean, and concise **custom web interface**.

---

## ✨ Features

-   ⚡ **Fast and local**: Runs without cloud APIs.
-   🔐 **Privacy-first**: All processing happens on your machine.
-   🧠 **Mistral + Ollama** for efficient LLM inference.
-   💻 **Elegant Web UI**: A custom-built, responsive web interface (HTML, CSS, JavaScript, Bootstrap 5) for a superior user experience, including secure authentication.
-   **User Authentication**: Robust Login and Sign-up pages for secure access and personalized interactions.
-   🧠 **Multimodal Intelligence**: Combines Faster-Whisper for transcription and Deep-Translation for multilingual support for sentiment analysis.
-   📦 Modular architecture for easy integration or expansion.

---

## 🌐 Web Interface (Frontend)

Beyond the powerful backend, IntelliTube now boasts a custom-built, responsive, and visually appealing frontend for a seamless user experience.

-   **Modern Design:** Built with **HTML, CSS, and Bootstrap 5**, offering a clean and intuitive layout.
-   **Dynamic Interactions:** Enhanced with **JavaScript** for interactive elements, including password strength validation and form handling.
-   **Authentication System:** Secure and user-friendly **Login and Sign-up pages** to manage user access.
-   **Responsive Layout:** Optimized for various screen sizes, from desktops to mobile devices.

---

## 🚀 Getting Started

### 1. Clone the Main Repo

```bash
git clone https://github.com/AtharSayed/CodeLLM.git
cd CodeLLM/Intellitube
