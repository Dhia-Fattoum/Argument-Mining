
import re

# Basic list of vague terms and modal expressions
VAGUE_TERMS = [
    "some", "many", "few", "often", "sometimes", "probably", "possibly",
    "might", "could", "may", "maybe", "it", "they", "that", "unclear", "ambiguous"
]
MODAL_VERBS = ["might", "could", "would", "should", "may"]

def is_ambiguous(text: str) -> bool:
    """
    Returns True if the sentence is considered ambiguous based on heuristics.
    """
    lower_text = text.lower()

    vague_found = [word for word in VAGUE_TERMS if re.search(rf"\b{word}\b", lower_text)]
    modal_found = [verb for verb in MODAL_VERBS if re.search(rf"\b{verb}\b", lower_text)]

    # Heuristic: if it contains multiple vague indicators, it's likely ambiguous
    return len(vague_found + modal_found) >= 2

def check_ambiguity(text: str) -> str:
    if is_ambiguous(text):
        return "⚠️ This sentence may be ambiguous. It contains vague or unclear terms."
    else:
        return "✅ This sentence appears to be clear and specific."

# Local test
if __name__ == "__main__":
    user_input = input("Enter a sentence to check for ambiguity:\n")
    result = check_ambiguity(user_input)
    print("\n" + result)