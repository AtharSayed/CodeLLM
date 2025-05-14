import whisper
import subprocess
import os
import torch

def download_and_transcribe_youtube(url, output="audio.mp3"):
    try:
        print("📥 Downloading audio...")
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3", "-o", output, url
        ], check=True)

        print("🧠 Transcribing audio...")
        model = whisper.load_model("base").to("cuda" if torch.cuda.is_available() else "cpu")
        result = model.transcribe(output)
        return result["text"]
    except Exception as e:
        print("❌ Error:", e)
        return "Failed to process video."
