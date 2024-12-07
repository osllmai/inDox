import json
from typing import List, Dict


class RougeTemplate:
    @staticmethod
    def analyze_rouge_scores(generated: str, reference: str) -> str:
        return f"""Analyze the ROUGE scores between the generated and reference summaries, considering:
1. ROUGE-1 (Unigram overlap)
2. ROUGE-2 (Bigram overlap)
3. ROUGE-L (Longest Common Subsequence)

For each metric, provide precision, recall, and F1-score details.

IMPORTANT: Return only in JSON format.
Example:
{{
    "scores": [
        {{
            "metric": "rouge_1",
            "score": {{
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81
            }},
            "details": {{
                "matching_unigrams": 45,
                "total_generated_unigrams": 53,
                "total_reference_unigrams": 58
            }},
            "reason": "Strong unigram overlap with good balance of precision and recall"
        }}
    ]
}}

Generated Summary:
{generated}

Reference Summary:
{reference}

JSON:"""

    @staticmethod
    def generate_final_verdict(scores: List[Dict], final_score: float) -> str:
        return f"""Based on the ROUGE scores analysis, provide a concise overall assessment of the summary's overlap quality.

Scores:
{json.dumps(scores, indent=2)}

Final ROUGE Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary shows strong lexical overlap with the reference, particularly in unigram matches, though bigram continuity could be improved."
}}

JSON:"""
