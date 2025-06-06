from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# Supported language pairs
lang_pair_model_map = {
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("zh-cn", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh-cn"): "Helsinki-NLP/opus-mt-en-zh",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("tr", "en"): "Helsinki-NLP/opus-mt-tr-en",
    ("en", "tr"): "Helsinki-NLP/opus-mt-en-tr",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("en", "nl"): "Helsinki-NLP/opus-mt-en-nl",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-ja"
}

def translate_text(text: str, target_lang: str = None):
    try:
        source_lang = detect(text)

        # If source is not English and no target provided → translate to English
        if target_lang is None and source_lang != "en":
            target_lang = "en"

        # If text is English but no target provided, return error
        if target_lang is None and source_lang == "en":
            return {
                "error": "Target language required when input is already English."
            }

        if source_lang == target_lang:
            return {
                "message": "Input and target language are the same.",
                "text": text
            }

        model_name = lang_pair_model_map.get((source_lang, target_lang))
        if not model_name:
            return {
                "error": f"No model found for translating from {source_lang} to {target_lang}."
            }

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translated_text": translated_text
        }

    except Exception as e:
        return {"error": str(e)}
