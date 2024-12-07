import json
from typing import Dict


class BleuTemplate:
    @staticmethod
    def generate_final_verdict(score: Dict, final_score: float) -> str:
        return f"""Based on the BLEU score analysis, provide a concise overall assessment of the summary's quality.

Scores:
{json.dumps(score, indent=2)}

Final BLEU Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary demonstrates good precision across n-grams, with particularly strong unigram matches. The brevity penalty indicates appropriate length alignment with the reference."
}}

JSON:"""
