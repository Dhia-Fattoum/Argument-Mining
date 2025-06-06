from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from datetime import datetime

from app.api.translate_route import router as translate_router
from app.api.sentiment import router as sentiment_router
from app.api.spellcheck import router as spellcheck_router
from app.api.toxicity import router as toxicity_router
from app.api.ambiguity import router as ambiguity_router
from app.api.argument import router as argument_router
from app.api.offensive import router as offensive_router
from app.models.speech_to_text import AdvancedSpeechToText
from app.api.argument_strength_polarity import router as arg_strength_router

# Initialize app and database
app = FastAPI()

# Register routers
app.include_router(translate_router)
app.include_router(sentiment_router)
app.include_router(spellcheck_router)
app.include_router(toxicity_router)
app.include_router(ambiguity_router)
app.include_router(argument_router)
app.include_router(offensive_router)
app.include_router(arg_strength_router)

# Speech-to-text setup
stt = AdvancedSpeechToText()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = stt.process_audio(filepath, language="de")
        return JSONResponse(content={"status": "ok", "results": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Politaktiv backend!"}
