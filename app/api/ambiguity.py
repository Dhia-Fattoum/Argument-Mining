from fastapi import APIRouter
from pydantic import BaseModel
from app.models.ambiguity_detector import is_ambiguous, explain_ambiguity

router = APIRouter()

class AmbiguityRequest(BaseModel):
    text: str

@router.post("/check")
def check_ambiguity(data: AmbiguityRequest):
    return {
        "ambiguous": is_ambiguous(data.text),
        "explanation": explain_ambiguity(data.text)
    }
