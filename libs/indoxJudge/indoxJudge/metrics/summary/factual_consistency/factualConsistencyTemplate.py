from typing import List, Dict
import json


class FactualConsistencyTemplate:
    @staticmethod
    def extract_claims(text: str) -> str:
        return f"""Extract all factual claims from the text and categorize them into:
1. Numerical Claims (statistics, measurements, dates) - use category "numerical_claims"
2. Entity Claims (about people, organizations, places) - use category "entity_claims"
3. Causal Claims (cause-effect relationships) - use category "causal_claims"
4. Descriptive Claims (properties, characteristics) - use category "descriptive_claims"
5. Comparative Claims (comparisons, rankings) - use category "comparative_claims"

IMPORTANT: Use exactly these category names in your response. Return only in JSON format.

Example:
{{
    "claims": [
        {{
            "category": "numerical_claims",
            "claim": "The study involved 500 participants",
            "context": "Sample size description in methodology",
            "text_span": "involved 500 participants"
        }}
    ]
}}

Text to analyze:
{text}

JSON:"""

    @staticmethod
    def verify_claims(summary_claims: List[Dict], source_text: str) -> str:
        claims_json = json.dumps(summary_claims, indent=2)
        return f"""Verify each claim from the summary against the source text. For each claim:
1. Assess factual consistency (0.0-1.0):
   - Score 1.0 for exact matches or semantically identical content
   - Score 0.9-0.99 for minor wording differences with same meaning
   - Score 0.7-0.89 for mostly consistent with slight variations
   - Score 0.5-0.69 for partially consistent with notable differences
   - Score 0.0-0.49 for inconsistent or unsupported claims

2. Find supporting evidence using exact text matching when possible
3. Identify any discrepancies or errors
4. Provide detailed explanation

Error types should only be assigned if there's a genuine discrepancy:
- contradiction: Direct conflict with source
- exaggeration: Clear overstatement of facts
- misattribution: Incorrect source or attribution
- unsupported: No supporting evidence found
- oversimplification: Significant loss of accuracy

IMPORTANT: Return only in JSON format.
Example:
{{
    "verified_claims": [
        {{
            "claim": "The study involved 500 participants",
            "source_evidence": "The study involved 500 participants",
            "consistency_score": 1.0,
            "error_type": null,
            "explanation": "Exact match found in source text"
        }},
        {{
            "claim": "The study showed a 50% improvement",
            "source_evidence": "The study demonstrated a 48% improvement rate",
            "consistency_score": 0.9,
            "error_type": "minor_exaggeration",
            "explanation": "Slight overstatement but essentially accurate"
        }}
    ]
}}

Claims to verify:
{claims_json}

Source Text:
{source_text}

JSON:"""

    @staticmethod
    def generate_category_verdict(verified_claims: List[Dict]) -> str:
        categories = [
            "numerical_claims",
            "entity_claims",
            "causal_claims",
            "descriptive_claims",
            "comparative_claims",
        ]

        return f"""Analyze the verified claims and provide a verdict for each category.
    Use exactly these category names:
    {', '.join(categories)}

    IMPORTANT: You must include a verdict for ALL categories listed above, even if no claims were found for that category (use score 1.0 for empty categories).

    Calculate scores based on:
    1. Average consistency scores for each category
    2. Proportion of fully consistent claims
    3. Impact of any errors on overall reliability

    All fields must match this format:
    - "consistent_claims": List of strings (one claim per entry)
    - "inconsistent_claims": List of dictionaries (each with "claim" and "reason")

    Example:
    {{
        "scores": [
            {{"category": "numerical_claims",
            "score": 1.0,
            "consistent_claims": ["The study had 500 participants"],
            "inconsistent_claims": [],
            "reason": "All numerical claims exactly match source text"}},
            {{"category": "entity_claims",
            "score": 1.0,
            "consistent_claims": [],
            "inconsistent_claims": [],
            "reason": "No entity claims found in the text"}}
        ]
    }}

    Verified Claims:
    {json.dumps(verified_claims, indent=2)}

    JSON:"""
