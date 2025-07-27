from youtube_comment_downloader import YoutubeCommentDownloader
from langdetect import detect
from deep_translator import GoogleTranslator
import re

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def is_english(text):
    try:
        lang = detect(text)
        return lang == "en"
    except:
        return False  # assume non-English if detection fails

def translate_to_english(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        return f"[Translation error: {str(e)}] {text}"

def fetch_comments_scrape(url, max_comments=500):
    video_id = extract_video_id(url)
    downloader = YoutubeCommentDownloader()
    comments = []

    try:
        for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
            text = comment["text"]
            if not is_english(text):
                text = translate_to_english(text)
            comments.append(text)
            if len(comments) >= max_comments:
                break
    except Exception as e:
        return [f"❌ Error fetching comments: {str(e)}"]

    return comments
