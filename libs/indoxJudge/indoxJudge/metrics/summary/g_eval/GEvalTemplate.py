import json
from typing import Dict, List


class GEvalTemplate:
    @staticmethod
    def evaluate_grammar(summary: str) -> str:
        return f"""Analyze the grammatical quality of the following text considering these aspects:
1. Grammar Correctness (Proper grammar rules, verb agreement, tense consistency)
2. Sentence Structure (Formation, complexity, and variety)
3. Coherence (Logical flow and connection between ideas)
4. Readability (Clarity and ease of understanding)

For each aspect, provide:
- A score between 0.0 and 1.0
- A reason for the score
- For Grammar Correctness, include a list of issues found

IMPORTANT: Return only in JSON format with scores as a list of objects.
Example response:
{{
    "scores": [
        {{
            "aspect": "Grammar Correctness",
            "score": 0.85,
            "reason": "Generally correct with minor issues in verb agreement and article usage",
            "issues_found": ["subject-verb disagreement in sentence 2", "missing article before noun"]
        }},
        {{
            "aspect": "Sentence Structure",
            "score": 0.90,
            "reason": "Well-constructed sentences with good variety in length and complexity"
        }},
        {{
            "aspect": "Coherence",
            "score": 0.88,
            "reason": "Ideas flow logically with clear transitions between paragraphs"
        }},
        {{
            "aspect": "Readability",
            "score": 0.92,
            "reason": "Text is clear and easy to understand with appropriate vocabulary"
        }}
    ]
}}

Text to analyze:
{summary}

JSON:"""

    @staticmethod
    def extract_grammar_issues(text: str) -> str:
        return f"""Identify all grammatical issues in the following text. Focus on:
1. Grammar rule violations
2. Subject-verb agreement
3. Tense consistency
4. Article usage
5. Pronoun reference
6. Punctuation
7. Sentence fragments or run-ons

Return only in JSON format with a 'grammar_issues' list.
Example:
{{
    "grammar_issues": [
        "Incorrect subject-verb agreement in first paragraph",
        "Missing article before noun 'project'",
        "Run-on sentence in conclusion",
        "Inconsistent verb tense in second paragraph",
        "Incorrect punctuation in dialogue"
    ]
}}

Text:
{text}

JSON:"""

    @staticmethod
    def generate_final_verdict(
        scores: List[Dict], final_score: float, issue_distribution: Dict[str, int]
    ) -> str:
        return f"""Generate a comprehensive assessment of the text's grammatical quality based on the following analysis:

Evaluation Scores:
{json.dumps(scores, indent=2)}

Distribution of Grammar Issues:
{json.dumps(issue_distribution, indent=2)}

Overall Grammar Score: {final_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The text demonstrates strong grammatical proficiency (score: 0.85) with particular strengths in sentence structure and readability. Minor improvements could be made in subject-verb agreement and article usage. The consistent flow and varied sentence structure contribute to effective communication, though attention to technical grammar rules would enhance overall quality."
}}

JSON:"""
