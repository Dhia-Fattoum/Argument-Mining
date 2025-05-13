# app/api/sentiment.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.sentiment import analyze_sentiment

router = APIRouter()

class SentimentInput(BaseModel):
    text: str

@router.post("/")
def sentiment_analysis(data: SentimentInput):
    result = analyze_sentiment(data.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
