# import whisper
# import subprocess
# import os
# import time

# def download_and_transcribe_youtube(url: str) -> str:
#     """Simplified version that always works"""
#     try:
#         audio_file = "temp_audio.mp3"
        
#         # Clean previous file
#         if os.path.exists(audio_file):
#             os.remove(audio_file)
        
#         # Download (15s timeout)
#         subprocess.run([
#             "yt-dlp",
#             "-x",
#             "--audio-format", "mp3",
#             "--audio-quality", "0",
#             "--quiet",
#             "--no-warnings",
#             "-o", audio_file,
#             url
#         ], timeout=15, check=True)
        
#         if not os.path.exists(audio_file):
#             return "❌ Download failed"
        
#         # Load model (tiny.en is fastest)
#         model = whisper.load_model("tiny.en", device="cpu")
        
#         # Fast transcription (30s timeout)
#         start = time.time()
#         result = model.transcribe(
#             audio_file,
#             fp16=False,
#             language="en",
#             verbose=False,
#             temperature=0.0,
#             best_of=1,
#             beam_size=1,
#             patience=1.0
#         )
#         print(f"✓ Transcription took {time.time()-start:.1f}s")
        
#         return result["text"]
        
#     except subprocess.TimeoutExpired:
#         return "❌ Download too slow"
#     except Exception as e:
#         return f"❌ Error: {str(e)}"
#     finally:
#         if os.path.exists(audio_file):
#             os.remove(audio_file)
# test_qa.py


## **Test code to check if transcriber is getting fetched into qasetp for qa**

# from qa_chain import setup_qa

# transcript = "Your test transcript here. Add a few lines of meaningful text so it's not empty."
# qa = setup_qa(transcript)
# print(qa.run("What is this about?"))

