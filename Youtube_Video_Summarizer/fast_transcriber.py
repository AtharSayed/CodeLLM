from faster_whisper import WhisperModel
import subprocess
import os

def download_audio(url):
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--quiet", "-o", "audio.mp3", url
    ], check=True)

def transcribe_audio():
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe("audio.mp3", beam_size=1)
    return " ".join(segment.text for segment in segments)

def transcribe_youtube(url):
    try:
        download_audio(url)
        return transcribe_audio()
    except Exception as e:
        return f"❌ Error: {str(e)}"
    finally:
        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")