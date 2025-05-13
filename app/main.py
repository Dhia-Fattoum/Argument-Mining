from fastapi import FastAPI
from app.api.translate_route import router as translate_router
from app.api.sentiment import router as sentiment_router
from app.api.toxicity import router as toxicity_router
from app.api.ambiguity import router as ambiguity_router
from app.api.spellcheck import router as spellcheck_router
from app.api.argument import router as argument_router
from app.api.argument_strength_polarity import router as arg_strength_router

app = FastAPI(title="Politaktiv AI Backend")

app.include_router(translate_router, prefix="/translate", tags=["Translation"])
app.include_router(sentiment_router, prefix="/sentiment", tags=["Sentiment"])
app.include_router(toxicity_router, prefix="/toxicity", tags=["Toxicity"])
app.include_router(ambiguity_router, prefix="/ambiguity", tags=["Ambiguity"])
app.include_router(spellcheck_router, prefix="/spellcheck", tags=["SpellCheck"])
app.include_router(argument_router, prefix="/argument", tags=["Argument"]) 
app.include_router(arg_strength_router, prefix="/argument-quality", tags=["Argument Strength & Polarity"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Politaktiv backend!"}
