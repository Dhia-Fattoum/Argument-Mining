from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# Supported (source, target) language pairs mapped to Helsinki-NLP models
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

def translate_text(text, target_lang=None):
    try:
        source_lang = detect(text)
        print(f"🌍 Detected Source Language: {source_lang}")
        
        # If no target specified and source isn't English, default to English
        if target_lang is None and source_lang != "en":
            target_lang = "en"
        elif target_lang is None and source_lang == "en":
            target_lang = input("Enter target language code (e.g., 'ar', 'fr', 'es'): ")
        
        if source_lang == target_lang:
            return "✅ Input and target languages are the same."

        model_name = lang_pair_model_map.get((source_lang, target_lang))

        if not model_name:
            return f"❌ No model found for translating from {source_lang} to {target_lang}."

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ✍️ Try it out:
#text = input("Enter text to translate: ")

# For non-English text, we won't ask for target language (defaults to English)
# For English text, we'll ask what language to translate to
#result = translate_text(text)
#print("\n🗽 Translated Text:")
#print(result)