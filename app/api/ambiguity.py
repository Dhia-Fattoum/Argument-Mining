from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.models.ambiguity_detector import check_ambiguity

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/ambiguity/check")
async def ambiguity_check(request: TextRequest):
    return check_ambiguity(request.text)
