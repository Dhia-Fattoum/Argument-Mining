from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.toxic_blocker import analyze_toxicity

router = APIRouter()

class ToxicityRequest(BaseModel): 
    text: str

@router.post("/toxic")
def check_toxicity(data: ToxicityRequest):
    result = analyze_toxicity(data.text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
