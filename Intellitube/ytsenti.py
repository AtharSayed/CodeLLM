import re
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from ytcom import fetch_comments_scrape
import numpy as np
import nltk
nltk.download('punkt')

# Device setup
device = 0 if torch.cuda.is_available() else -1
print("Device set to:", "cuda:0" if device == 0 else "cpu")

# Updated sentiment model (Roberta-based)
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=device)

# Emotion/Intent model
intent_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    device=device
)

# Custom label mappings
SENTIMENT_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

INTENT_MAP = {
    "joy": "Praise",
    "anger": "Complaint",
    "sadness": "Disappointment",
    "surprise": "Curiosity",
    "love": "Appreciation",
    "fear": "Concern"
}

def preprocess(text):
    return re.sub(r"http\S+|www.\S+", "", text).strip()

def analyze_sentiment(comments):
    print("✅ Performing semantic sentiment analysis...")
    summary = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    detailed = []

    for comment in comments:
        try:
            cleaned = preprocess(comment)
            result = sentiment_analyzer(cleaned)[0]
            label = SENTIMENT_MAP.get(result["label"], "NEUTRAL")
            score = result["score"]

            summary[label] += 1
            detailed.append((comment, label, score))
        except Exception as e:
            detailed.append((comment, "ERROR", 0.0))

    return summary, detailed


def analyze_intent(comments):
    print("✅ Performing semantic intent classification...")
    intent_summary = {}
    intent_detailed = []

    for comment in comments:
        try:
            cleaned = preprocess(comment)
            result = intent_classifier(cleaned)[0]
            raw_label = result["label"]
            score = result["score"]
            label = INTENT_MAP.get(raw_label, raw_label)

            intent_summary[label] = intent_summary.get(label, 0) + 1
            intent_detailed.append((comment, label, score))
        except Exception:
            intent_detailed.append((comment, "ERROR", 0.0))

    return intent_summary, intent_detailed


def display_results(sent_summary, intent_summary, sent_detailed, intent_detailed):
    print("\n--- Sentiment Summary ---")
    for sentiment, count in sent_summary.items():
        emoji = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(sentiment, "🔘")
        print(f"{emoji} {sentiment}: {count}")

    overall = max(sent_summary, key=sent_summary.get)
    print(f"📊 Overall Sentiment: Mostly {overall.capitalize()}")

    print("\n--- Intent Summary ---")
    for label, count in intent_summary.items():
        print(f"🔹 {label}: {count}")

    print("\n📋 Detailed Sentiment + Intent Analysis:")
    for i, (comment, sentiment, s_score) in enumerate(sent_detailed, 1):
        intent_entry = next(((cmt, intent, i_score) for cmt, intent, i_score in intent_detailed if cmt == comment), None)
        intent_label = intent_entry[1] if intent_entry else "Unknown"
        print(f"{i}. {comment}")
        print(f"   🧠 Sentiment: {sentiment} ({round(s_score, 3)}), 🎯 Intent: {intent_label}")

def main():
    url = "https://www.youtube.com/watch?v=QwievZ1Tx-8&themeRefresh=1"

    comments = fetch_comments_scrape(url, max_comments=50)
    if not comments:
        print("❌ No comments found.")
        return

    sentiment_summary, sentiment_detailed = analyze_sentiment(comments)
    intent_summary, intent_detailed = analyze_intent(comments)

    display_results(sentiment_summary, intent_summary, sentiment_detailed, intent_detailed)


# --> Uncomment only for testing sentiment Analysis Functionality 
# if __name__ == "__main__":
#     main()
