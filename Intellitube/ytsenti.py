import re
from youtube_comment_downloader import YoutubeCommentDownloader
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

# Sentiment pipeline setup
device = 0 if torch.cuda.is_available() else -1
print("Device set to:", "cuda:0" if device == 0 else "cpu")
sentiment_analyzer = pipeline("sentiment-analysis", device=device)

# Extracts YouTube video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

# Language detection
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

# Translate if not English
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return f"[Translation error] {text}"

# Fetch YouTube comments
def fetch_comments_scrape(url, max_comments=50):
    video_id = extract_video_id(url)
    downloader = YoutubeCommentDownloader()
    comments = []
    print("💬 Fetching comments...")

    try:
        for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
            text = comment["text"]
            if not is_english(text):
                text = translate_to_english(text)
            comments.append(text)
            if len(comments) >= max_comments:
                break
    except Exception as e:
        print(f"❌ Error fetching comments: {e}")
        return []

    return comments

# Perform sentiment analysis
def analyze_sentiment(comments):
    print("✅ Performing sentiment analysis...")
    summary = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    detailed = []

    for comment in comments:
        try:
            result = sentiment_analyzer(comment)[0]
            label = result["label"]
            score = result["score"]

            if label == "POSITIVE":
                summary["POSITIVE"] += 1
            elif label == "NEGATIVE":
                summary["NEGATIVE"] += 1
            else:
                summary["NEUTRAL"] += 1

            detailed.append((comment, label, score))
        except Exception as e:
            detailed.append((comment, "ERROR", 0.0))

    return summary, detailed

# Main entry point
def main():
    url = input("🔗 Enter YouTube video URL: ").strip()
    if not url:
        print("❌ No URL entered.")
        return

    comments = fetch_comments_scrape(url, max_comments=50)
    if not comments:
        print("❌ No comments to analyze.")
        return

    summary, detailed = analyze_sentiment(comments)

    print("\n--- Sentiment Analysis Summary ---")
    print("🟢 POSITIVE:", summary["POSITIVE"])
    print("🟡 NEUTRAL :", summary["NEUTRAL"])
    print("🔴 NEGATIVE:", summary["NEGATIVE"])

    if summary["POSITIVE"] > summary["NEGATIVE"]:
        print("✅ Overall Sentiment: Mostly Positive")
    elif summary["NEGATIVE"] > summary["POSITIVE"]:
        print("⚠️ Overall Sentiment: Mostly Negative")
    else:
        print("📊 Overall Sentiment: Mixed or Neutral")

    print("\n📋 Detailed Comment Analysis:")
    for i, (comment, label, score) in enumerate(detailed, 1):
        print(f"{i}. {comment}")
        print(f"   Sentiment: {label}, Score: {round(score, 3)}")

if __name__ == "__main__":
    main()
