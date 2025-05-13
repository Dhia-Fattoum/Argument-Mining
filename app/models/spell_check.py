from spellchecker import SpellChecker
from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoTokenizer,
    pipeline
)
import torch
import re

class MegaSpellCorrector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use smaller models for efficiency
        self.models = {
            "bart": {
                "model": BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(self.device),
                "tokenizer": AutoTokenizer.from_pretrained("facebook/bart-base")
            },
            "t5": {
                "model": T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device),
                "tokenizer": AutoTokenizer.from_pretrained("t5-small")
            }
        }

        self.spell = SpellChecker()
        try:
            self.grammar_fixer = pipeline(
                "text2text-generation",
                model="pszemraj/flan-t5-large-grammar-synthesis",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # Fallback model
            self.grammar_fixer = pipeline(
                "text2text-generation",
                model="vennify/t5-base-grammar-correction",
                device=0 if torch.cuda.is_available() else -1
            )

        for model in self.models.values():
            model["model"].eval()

    def _generate_candidates(self, text, model_name):
        try:
            model = self.models[model_name]
            inputs = model["tokenizer"](
                [f"grammar correction: {text}"],  # Simpler prompt
                max_length=512,  # Reduced max length
                return_tensors="pt",
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = model["model"].generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_beams=3,  # Fewer beams for speed
                    early_stopping=True
                )

            return model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in {model_name} generation: {str(e)}")
            return text  # Return original text if error occurs

    def correct_text(self, text):
        if not text.strip():
            return text
        
        try:
            # First apply grammar correction
            corrected = self.grammar_fixer(text)[0]['generated_text']
            
            # Tokenize into words
            words = corrected.split()
            corrected_words = []

            for word in words:
                # Strip punctuation
                word_clean = re.sub(r'[^\w]', '', word)
                if not word_clean or not word_clean.isalpha():
                    corrected_words.append(word)
                    continue
                
                correction = self.spell.correction(word_clean)
                corrected_word = correction if correction else word_clean

                # Preserve capitalization
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                    
                corrected_words.append(corrected_word)

            result = ' '.join(corrected_words)
            
            # Apply grammar correction again
            return self.grammar_fixer(result)[0]['generated_text']

        except Exception as e:
            print(f"Error in correction: {str(e)}")
            return text

# Singleton instance for MegaSpellCorrector
corrector_instance = MegaSpellCorrector()

def correct_spelling(text: str) -> str:
    return corrector_instance.correct_text(text)
    