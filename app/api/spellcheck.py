from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.spell_check import correct_spelling  # Adjust the import path if needed

router = APIRouter()

class SpellCheckRequest(BaseModel):
    text: str 

@router.post("/correct")
def spell_check(request: SpellCheckRequest):
    try:
        corrected = correct_spelling(request.text)
        return {"original": request.text, "corrected": corrected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   