from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.toxic_blocker import analyze_toxicity 

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/offensive")
def detect_toxicity(input: TextInput):
    # Call analyze_toxicity which returns a dictionary with 'scores' and 'is_offensive'
    result = analyze_toxicity(input.text)
    
    if "error" in result:
        # Handle potential errors from the model analysis
        raise HTTPException(status_code=500, detail=f"Toxicity analysis error: {result['error']}")

    # Return the toxicity scores directly
    return {
        "text": input.text,
        "is_offensive": result["is_offensive"],
        "toxicity_scores": result["scores"] # This will already be a dict of floats, no .tolist() needed
    }