import whisper
import subprocess
import torch

def download_and_transcribe_youtube(url, output="audio.mp3"):
    try:
        print("📥 Downloading audio...")
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3", "-o", output, url
        ], check=True)

        # Using CPU to run the Transcribing model 
        device = "cpu" 
        print(f"🧠 Transcribing using device: {device} ")

        # Load the Whisper model on the correct device
        model = whisper.load_model("tiny", device=device)

        # Transcribe audio
        result = model.transcribe(output)
        return result["text"]

    except Exception as e:
        error_msg = f"❌ Failed to process video. Error: {str(e)}"
        print(error_msg)
        return error_msg
