import whisper
import subprocess
import os
import time
from typing import Optional
from functools import lru_cache

@lru_cache(maxsize=1)
def load_whisper_model():
    """Cache the loaded model to avoid reloading"""
    print("⚙️ Loading Whisper model...")
    start = time.time()
    model = whisper.load_model(
        "tiny",
        device="cpu",
        download_root="./whisper_models"
    )
    print(f"✅ Model loaded in {time.time()-start:.1f}s")
    return model

def download_audio(url: str, output: str, timeout: int = 60) -> bool:
    """Download audio with timeout handling"""
    try:
        print("⬇️ Downloading audio...")
        start = time.time()
        subprocess.run([
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--quiet",
            "-o", output,
            url
        ], check=True, timeout=timeout)
        print(f"✅ Download completed in {time.time()-start:.1f}s")
        return True
    except subprocess.TimeoutExpired:
        print(f"⌛ Download timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

def transcribe_audio(file_path: str, timeout: int = 120) -> Optional[str]:
    """Transcribe with timeout protection"""
    try:
        print("🔊 Starting transcription...")
        start = time.time()
        
        # Use event to handle timeout
        from threading import Event, Thread
        result = {}
        event = Event()
        
        def _transcribe():
            try:
                model = load_whisper_model()
                result['text'] = model.transcribe(
                    file_path,
                    fp16=False,
                    language="en",
                    verbose=True,
                    temperature=0.0,
                    best_of=1,
                    beam_size=1
                )["text"]
            except Exception as e:
                result['error'] = str(e)
            finally:
                event.set()
        
        thread = Thread(target=_transcribe)
        thread.start()
        event.wait(timeout=timeout)
        
        if not event.is_set():
            raise TimeoutError(f"Transcription timed out after {timeout}s")
            
        if 'error' in result:
            raise Exception(result['error'])
            
        print(f"✅ Transcription completed in {time.time()-start:.1f}s")
        return result['text']
        
    except Exception as e:
        print(f"❌ Transcription failed: {str(e)}")
        return None

def download_and_transcribe_youtube(url: str, output: str = "audio.mp3") -> str:
    """Main processing pipeline with timeout protection"""
    try:
        # Clean previous files
        if os.path.exists(output):
            os.remove(output)
        
        # Download with timeout
        if not download_audio(url, output):
            return "❌ Audio download failed"
        
        if not os.path.exists(output):
            return "❌ Downloaded file not found"
        
        # Transcribe with timeout
        text = transcribe_audio(output)
        if text is None:
            return "❌ Transcription failed"
        
        # Clean up
        os.remove(output)
        return text
        
    except Exception as e:
        if os.path.exists(output):
            os.remove(output)
        return f"❌ Processing failed: {str(e)}"