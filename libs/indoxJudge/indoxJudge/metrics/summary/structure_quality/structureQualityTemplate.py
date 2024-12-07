from typing import List, Dict
import json


class StructureTemplate:
    @staticmethod
    def evaluate_structure(summary: str) -> str:
        return f"""Analyze the structural quality of the following summary by evaluating these aspects:
1. Discourse coherence (How well sentences and ideas connect)
2. Logical flow (Natural progression of ideas)
3. Topic consistency (Staying on relevant subjects)
4. Temporal consistency (Proper time sequence if applicable)

For each aspect, provide a score between 0.0 and 1.0 and a brief reason.

IMPORTANT: Return only in JSON format with scores as a list of objects containing 'aspect', 'score', and 'reason'.
Example response:
{{
    "scores": [
        {{
            "aspect": "discourse_coherence",
            "score": 0.85,
            "reason": "Sentences flow naturally with clear connections using appropriate transition words"
        }},
        {{
            "aspect": "logical_flow",
            "score": 0.90,
            "reason": "Ideas progress from general to specific with clear cause-effect relationships"
        }}
    ]
}}

Summary to analyze:
{summary}

JSON:"""

    @staticmethod
    def generate_final_verdict(scores: List[Dict], final_score: float) -> str:
        return f"""Based on the structural analysis scores, provide a concise overall assessment of the summary's structural quality.

Scores:
{json.dumps(scores, indent=2)}

Final Structure Quality Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary exhibits strong structural quality with particularly good logical flow, though discourse coherence could be improved."
}}

JSON:"""
