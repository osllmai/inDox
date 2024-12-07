from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationWeights:
    """Predefined weights for different evaluation aspects"""

    SEMANTIC_SIMILARITY = {
        "BertScore": 0.15,  # Strong semantic similarity measure
        "Rouge": 0.12,  # Traditional summary evaluation metric
        "Meteor": 0.08,  # Good for paraphrase detection
        "Bleu": 0.05,  # Basic overlap metric
    }

    CONTENT_QUALITY = {
        "FactualConsistency": 0.15,  # Critical for accuracy
        "InformationCoverage": 0.12,  # Important for completeness
        "Relevance": 0.10,  # Ensures summary captures key points
    }

    STRUCTURE_QUALITY = {
        "StructureQuality": 0.08,  # Readable and well-organized
        "Conciseness": 0.07,  # Efficient information presentation
    }

    SAFETY_METRICS = {
        "Toxicity": 0.04,  # Ensure safe content
        "GEval": 0.04,  # General evaluation quality
    }

    @classmethod
    def get_all_weights(cls) -> Dict[str, float]:
        """Combine all weights into a single dictionary"""
        weights = {}
        for category in [
            cls.SEMANTIC_SIMILARITY,
            cls.CONTENT_QUALITY,
            cls.STRUCTURE_QUALITY,
            cls.SAFETY_METRICS,
        ]:
            weights.update(category)
        return weights
