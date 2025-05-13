# app/api/argument.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.models.argument_extractor import ArgumentExtractor

router = APIRouter()
extractor = ArgumentExtractor()

class ArgumentRequest(BaseModel):
    text: str

@router.post("/extract")
def extract_argument_type(request: ArgumentRequest):
    return extractor.extract_argument(request.text)
