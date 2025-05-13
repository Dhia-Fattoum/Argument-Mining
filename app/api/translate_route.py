from fastapi import APIRouter
from pydantic import BaseModel
from app.models.translate import translate_text

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str
    target_lang: str = None  # Optional field

@router.post("/text")
def translate_route(request: TranslationRequest):
    translation = translate_text(request.text, request.target_lang)
    return {"translation": translation}
