from fastapi import APIRouter
from pydantic import BaseModel
from app.models.argument_strength_polarity import classify_argument

router = APIRouter()

class ArgumentClassificationRequest(BaseModel):
    text: str

@router.post("/classify")
def classify_argument_endpoint(request: ArgumentClassificationRequest):
    return classify_argument(request.text)
