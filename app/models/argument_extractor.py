from transformers import pipeline
from typing import Dict, Union
import torch

class ArgumentExtractor:
    def __init__(self):
        """Initialize the zero-shot classifier pipeline."""
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

        self.argument_types = {
            "Thesis": ["claim", "opinion", "recommendation", "proposal", "suggestion"],
            "Justification": ["reason", "evidence", "supporting idea", "because", "goal"],
            "Counter-argument": ["opposing view", "rebuttal", "disagreement", "refutation", "objection"]
        }

    def extract_argument(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Classifies input text as one of the argument components.

        Returns:
            - label: Thesis / Justification / Counter-argument
            - score: confidence score (0-1)
            - note: fallback info (if used)
        """
        if not text.strip():
            return {"error": "Empty input text"}
        
        try:
            labels = list(self.argument_types.keys())
            result = self.classifier(text, labels)

            # Smart fallback if confidence is too low
            if result["scores"][0] < 0.6:
                return {
                    "label": "Thesis",
                    "score": round(result["scores"][0], 2),
                    "note": "Fallback to Thesis due to low confidence"
                }

            return {
                "label": result["labels"][0],
                "score": round(result["scores"][0], 2)
            }

        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
