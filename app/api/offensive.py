from fastapi import APIRouter
from pydantic import BaseModel
from app.models.offensiveness_blocker import check_offensiveness

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/detect-toxicity")
def detect_toxicity(input: TextInput):
    label, scores = check_offensiveness(input.text)
    return {
        "text": input.text,
        "toxicity_label": label,
        "toxicity_scores": scores.tolist()
    }
    