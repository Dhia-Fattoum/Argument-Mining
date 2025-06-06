from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi import HTTPException

MODEL_NAME = "unitary/toxic-bert"

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load offensive language detection model: {e}")

def is_offensive(text: str, threshold: float = 0.5) -> bool:
    """
    Checks whether the text is offensive based on the toxic probability score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        toxic_prob = probs[0][1].item()
        return toxic_prob >= threshold

def check_offensiveness(text: str):
    """
    Raises HTTPException if the text is offensive.
    """
    if is_offensive(text): # This calls the is_offensive function
        raise HTTPException(
            status_code=400,
            detail="Offensive language detected. Please rephrase your message."
        )
    return {"status": "ok", "message": "Message is appropriate."}