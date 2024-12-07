import json
from typing import Dict


class BertScoreTemplate:
    @staticmethod
    def analyze_bert_scores(generated: str, reference: str) -> str:
        return f"""Analyze the BERTScore between the generated and reference texts, considering:
1. Precision (how many tokens in the generated text have good matches in the reference)
2. Recall (how many tokens in the reference text have good matches in the generated)
3. F1 Score (harmonic mean of precision and recall)

IMPORTANT: Return only in JSON format. Example:
{{
    "scores": {{
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.81,
        "details": {{
            "avg_precision_similarity": 0.85,
            "avg_recall_similarity": 0.78,
            "token_matches": 45
        }},
        "reason": "Strong semantic similarity with good balance of precision and recall"
    }}
}}

Generated Text: {generated}
Reference Text: {reference}

JSON:"""

    @staticmethod
    def generate_final_verdict(score: Dict, final_score: float) -> str:
        return f"""Based on the BERTScore analysis, provide a concise overall assessment of the semantic similarity between the texts.

Scores: {json.dumps(score, indent=2)}
Final BERTScore (F1): {final_score:.2f}

Return only in JSON format with a 'verdict' key. Example:
{{
    "verdict": "The generated text demonstrates strong semantic alignment with the reference, showing high precision in word choice and good coverage of reference concepts."
}}

JSON:"""
