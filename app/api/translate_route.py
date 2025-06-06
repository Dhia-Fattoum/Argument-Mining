from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from app.models.translate import translate_text

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str
    target_lang: Optional[str] = None

@router.post("/text")
def translate(request: TranslateRequest):
    result = translate_text(request.text, request.target_lang)
    return result
