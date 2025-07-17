import re
from transformers import pipeline
import torch
from ytcom import fetch_comments_scrape

# Sentiment pipeline setup
device = 0 if torch.cuda.is_available() else -1
print("Device set to:", "cuda:0" if device == 0 else "cpu")
sentiment_analyzer = pipeline("sentiment-analysis", device=device)


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

# Intent classification pipeline
intent_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",  # Temp model: classify emotion/intent-like labels
    device=device
)
# Analyze intent of comments
def analyze_intent(comments):
    print("✅ Performing intent classification...")
    intent_summary = {}
    intent_detailed = []

    # Intent label mapping
    INTENT_MAP = {
        "joy": "Praise",
        "anger": "Complaint",
        "sadness": "Complaint",
        "surprise": "Request",
        "love": "Praise",
        "fear": "Concern",
    }

    for comment in comments:
        try:
            result = intent_classifier(comment)[0]
            raw_label = result["label"]
            score = result["score"]

            # Normalize label using intent map
            label = INTENT_MAP.get(raw_label, raw_label)

            # Count summary
            if label not in intent_summary:
                intent_summary[label] = 0
            intent_summary[label] += 1

            intent_detailed.append((comment, label, score))
        except Exception as e:
            intent_detailed.append((comment, "ERROR", 0.0))

    return intent_summary, intent_detailed


if __name__ == "__main__":
    main()