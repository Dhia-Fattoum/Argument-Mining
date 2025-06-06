
from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_argument_strength(text: str) -> dict:
    labels = ["Strong", "Weak"]
    result = classifier(text, labels)
    return {
        "strength": result["labels"][0],
        "confidence": round(result["scores"][0], 2)
    }

def classify_argument_polarity(text: str) -> dict:
    labels = ["Pro", "Con"]
    result = classifier(text, labels)
    return {
        "polarity": result["labels"][0],
        "confidence": round(result["scores"][0], 2)
    }

def classify_argument(text: str) -> dict:
    """
    Returns both strength and polarity classification.
    """
    try:
        strength_result = classify_argument_strength(text)
        polarity_result = classify_argument_polarity(text)
        return {
            "strength": strength_result,
            "polarity": polarity_result
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Test block
if __name__ == "__main__":
    user_input = input("Enter an argument to classify:\n")
    result = classify_argument(user_input)

    if "error" in result:
        print(f"\n⚠️ Error: {result['error']}")
    else:
        print(f"\n💪 Strength: {result['strength']['strength']} ({result['strength']['confidence']})")
        print(f"🟢 Polarity: {result['polarity']['polarity']} ({result['polarity']['confidence']})")
