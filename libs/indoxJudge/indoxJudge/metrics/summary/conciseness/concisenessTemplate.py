class ConcisenessTemplate:
    @staticmethod
    def analyze_redundancy(text: str) -> str:
        return f"""Analyze the text for redundancy and repetition. Identify:
1. Repeated phrases or concepts
2. Redundant information
3. Unnecessary modifiers or qualifiers

IMPORTANT: Return only in JSON format.
Example:
{{
    "repeated_phrases": [
        {{"phrase": "important factor", "count": 3}},
        {{"phrase": "significantly impacts", "count": 2}}
    ],
    "redundant_information": [
        "The same study results mentioned twice",
        "Restated conclusion in different words"
    ],
    "unnecessary_modifiers": [
        "very", "really", "basically"
    ],
    "score": 0.85,
    "explanation": "Minor redundancy in key phrases, some unnecessary modifiers"
}}

Text to analyze:
{text}

JSON:"""

    @staticmethod
    def measure_wordiness(text: str) -> str:
        return f"""Analyze the text for wordiness and complexity. Consider:
1. Sentence length and structure
2. Use of filler words
3. Complex phrases that could be simplified
4. Overall word economy

IMPORTANT: Return only in JSON format.
Example:
{{
    "total_words": 150,
    "average_sentence_length": 15.5,
    "filler_word_count": 8,
    "complex_phrase_count": 3,
    "score": 0.75,
    "suggestions": [
        "Replace 'due to the fact that' with 'because'",
        "Simplify 'in spite of the fact that' to 'although'"
    ]
}}

Text to analyze:
{text}

JSON:"""
