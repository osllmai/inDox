import json
from typing import Dict, List


class RelevanceTemplate:
    @staticmethod
    def evaluate_relevance(summary: str, source_text: str) -> str:
        return f"""Analyze the relevance of the summary compared to the source text by evaluating these aspects:
1. Key Information Coverage (How well essential points are captured)
2. Topic Alignment (How well the summary stays aligned with the main topics)
3. Information Accuracy (How accurately the information is presented)
4. Focus Distribution (Whether emphasis is placed on the right aspects)

For each aspect, provide:
- A score between 0.0 and 1.0
- A reason for the score
- For Key Information Coverage, include a list of matched key points

IMPORTANT: Return only in JSON format with scores as a list of objects.
Example response:
{{
    "scores": [
        {{
            "aspect": "key_information_coverage",
            "score": 0.85,
            "reason": "The summary captures most essential points but misses the discussion of X",
            "key_points_matched": ["point 1", "point 2", "point 3"]
        }},
        {{
            "aspect": "topic_alignment",
            "score": 0.90,
            "reason": "Summary maintains consistent focus on the main topics without deviation"
        }}
    ]
}}

Source Text:
{source_text}

Summary to analyze:
{summary}

JSON:"""

    @staticmethod
    def extract_key_points(text: str) -> str:
        return f"""Extract the key points from the following text. These will be used as a reference for evaluating summary relevance.

Return only in JSON format with a 'key_points' list.
Example:
{{
    "key_points": [
        "Main argument of the text",
        "Critical supporting evidence",
        "Key conclusions or findings"
    ]
}}

Text:
{text}

JSON:"""

    @staticmethod
    def generate_final_verdict(
        scores: List[Dict], final_score: float, key_points_coverage: Dict[str, bool]
    ) -> str:
        return f"""Based on the relevance analysis scores and key points coverage, provide a concise overall assessment.

Scores:
{json.dumps(scores, indent=2)}

Key Points Coverage:
{json.dumps(key_points_coverage, indent=2)}

Final Relevance Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary demonstrates strong relevance with excellent coverage of key points, particularly in [specific aspect], though [area for improvement] could be enhanced."
}}

JSON:"""
