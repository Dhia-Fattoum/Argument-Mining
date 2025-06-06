# toxic_blocker.py

from detoxify import Detoxify

# Thresholds per category for flagging as offensive
TOXICITY_THRESHOLDS = {
    "toxicity": 0.5,
    "severe_toxicity": 0.4,
    "obscene": 0.5,
    "threat": 0.3,
    "insult": 0.5,
    "identity_attack": 0.4,
    "sexual_explicit": 0.4
}

CATEGORY_DESCRIPTIONS = {
    "toxicity": "Offensive or rude language",
    "severe_toxicity": "Highly aggressive or violent",
    "obscene": "Profanity or lewd language",
    "threat": "Implied harm or violence",
    "insult": "Name-calling or degrading",
    "identity_attack": "Offense targeting identity",
    "sexual_explicit": "Sexually explicit content"
}

def analyze_toxicity(text: str) -> dict:
    """
    Analyzes the toxicity scores of a given text.
    Returns a dictionary with scores and an 'is_offensive' flag.
    """
    try:
        model = Detoxify("original")
        scores = model.predict(text)

        is_offensive = any(
            scores[category] > TOXICITY_THRESHOLDS.get(category, 0.5)
            for category in scores
        )

        return {
            "scores": {k: float(v) for k, v in scores.items()},
            "is_offensive": is_offensive
        }

    except Exception as e:
        return {
            "error": str(e),
            "is_offensive": False,
            "scores": {}
        }

# Optional: Local test
if __name__ == "__main__":
    user_input = input("Enter text to check for toxicity:\n")
    result = analyze_toxicity(user_input)

    if "error" in result:
        print(f"⚠️ Error: {result['error']}")
    else:
        print("\n🧪 Toxicity Scores:")
        for category, score in result["scores"].items():
            label = category.replace("_", " ").title()
            print(f"{label}: {score:.2f} – {CATEGORY_DESCRIPTIONS.get(category, '')}")

        if result["is_offensive"]:
            print("\n⚠️ Warning: The text may contain offensive content!")
        else:
            print("\n✅ The text appears to be safe.")
