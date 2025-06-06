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
                word_clean = re.sub(r'[^\\w]', '', word)
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