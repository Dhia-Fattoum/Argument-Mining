from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os
import shutil
from app.models.speech_to_text import process_audio

router = APIRouter()

@router.post("/voice-text/") 
async def voice_to_text_api(
    file: UploadFile = File(...),
    enhance: bool = Query(True, description="Enhance transcription using Nous Hermes."),
    translate: bool = Query(False, description="Translate final result using DeepL."),
    target_lang: str = Query("en", description="Target language for translation.")
):
    try:
        # Save uploaded file to temp dir
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the voice processing pipeline
        result = process_audio(
            audio_path=temp_path,
            enhance=enhance,
            translate=translate,
            target_lang=target_lang
        )

        # Clean up
        os.remove(temp_path)

        return JSONResponse(content={"status": "success", "result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
 