# sentiment.py

from transformers import pipeline

# Load sentiment analysis pipeline once
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment of the input text.
    Returns a dictionary with label and confidence score.
    """
    try:
        result = sentiment_pipeline(text)[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 2)
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Test
if __name__ == "__main__":
    user_input = input("Enter a sentence to analyze sentiment:\n")
    sentiment = analyze_sentiment(user_input)

    if "error" in sentiment:
        print(f"\n⚠️ Error: {sentiment['error']}")
    else:
        emoji = {
            "POSITIVE": "😊",
            "NEGATIVE": "😠",
            "NEUTRAL": "😐"
        }.get(sentiment["label"], "")
        print(f"\n🧠 Sentiment: {sentiment['label']} {emoji}")
        print(f"Confidence: {sentiment['score']}")
