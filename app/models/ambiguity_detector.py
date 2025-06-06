# ambiguity_detector.py

from fastapi import HTTPException
from transformers import pipeline
import torch
import logging
import os # Import os module

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

print(f"--- DEBUG: Loading ambiguity_detector.py from: {os.path.abspath(__file__)} ---")

class AmbiguityDetector:
    def __init__(self):
        """
        Initializes the zero-shot classification pipeline for ambiguity detection.
        Uses facebook/bart-large-mnli. This ensures the model is loaded only once.
        """
        print("--- AmbiguityDetector: Initializing model (Version 2.1 - Model-Based, Adjusted Threshold) ---") # Console verification
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1 # Use GPU if available
            )
            logging.info("AmbiguityDetector model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ambiguity detection model: {e}")
            self.classifier = None # Set to None if loading fails to prevent further errors

        # Refined labels for zero-shot classification to better capture ambiguity.
        # These labels guide the NLI model to classify the sentence's nature.
        self.candidate_labels = [
            "The sentence has only one clear interpretation.",
            "The sentence has multiple possible interpretations or meanings."
        ]
        self.ambiguous_label = "The sentence has multiple possible interpretations or meanings."
        
        # This threshold is crucial for tuning. It determines how confident the model needs to be
        # in the "ambiguous" label to flag the sentence.
        self.confidence_threshold = 0.80 # <<<--- ADJUSTED THRESHOLD TO 0.80

    def is_ambiguous(self, text: str) -> bool:
        """
        Detects if the input text contains ambiguous language using a pre-trained model.
        Returns True if the text is classified as ambiguous with high confidence.
        """
        if not self.classifier:
            logging.error("Ambiguity detection model not initialized. Cannot process text.")
            return False # Default to non-ambiguous if model is not loaded

        if not text.strip():
            return False # Empty input is not ambiguous

        try:
            result = self.classifier(text, self.candidate_labels, multi_label=False)
            top_label = result['labels'][0]
            top_score = result['scores'][0]

            print(f"--- DEBUG: Ambiguity result for '{text[:20]}...': Top Label: '{top_label}', Score: {top_score:.2f} ---") # Debug print to console

            if top_label == self.ambiguous_label and top_score >= self.confidence_threshold:
                logging.info(f"Text flagged as AMBIGUOUS: '{text[:50]}...' (Score: {top_score:.2f}, Label: '{top_label}')")
                return True
            else:
                logging.info(f"Text classified as CLEAR: '{text[:50]}...' (Top label: '{top_label}', Score: {top_score:.2f})")
                return False

        except Exception as e:
            logging.error(f"Error during ambiguity detection for text: '{text[:50]}...'. Error: {e}")
            return False

# Instantiate the detector globally. This is vital.
ambiguity_detector_instance = AmbiguityDetector()

def check_ambiguity(text: str):
    """
    API endpoint function that uses the AmbiguityDetector instance.
    Raises HTTPException if the text is ambiguous, else returns an OK message.
    """
    if ambiguity_detector_instance.is_ambiguous(text):
        raise HTTPException(
            status_code=400,
            detail="This sentence may be ambiguous. It contains potentially unclear language."
        )
    else:
        # Updated success message with a clear version identifier
        return {"status": "ok", "message": "This sentence appears to be clear and specific (Ambiguity Model Version 2.1)."}

if __name__ == "__main__":
    print("\n--- Direct Script Testing: Ambiguity Detector (Version 2.1) ---")
    test_sentences = [
        "She made her duck.",  # Expected ambiguous
        "The bank was muddy.", # Expected ambiguous
        "You are good person.", # Expected clear now
        "The quick brown fox jumps over the lazy dog.", # Expected clear
    ]
    for sentence in test_sentences:
        print(f"\nDirectly testing: '{sentence}'")
        try:
            response = check_ambiguity(sentence)
            print(f"Direct Test Result: {response}")
        except HTTPException as e:
            print(f"Direct Test Result (HTTP Exception): Status Code {e.status_code}, Detail: {e.detail}")