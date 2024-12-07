import json
from typing import Dict


class MeteorTemplate:
    @staticmethod
    def generate_final_verdict(score: Dict, final_score: float) -> str:
        return f"""Based on the METEOR score analysis, provide a concise overall assessment of the summary's quality.

Scores:
{json.dumps(score, indent=2)}

Final METEOR Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary shows strong semantic alignment with the reference, particularly in exact matches and synonym usage, with good word order preservation."
}}

JSON:"""
