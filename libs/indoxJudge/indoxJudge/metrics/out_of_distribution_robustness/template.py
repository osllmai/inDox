class OODRobustnessTemplate:
    @staticmethod
    def generate_reason(input: str) -> str:
        return f"""Based on the given input, please carefully analyze the text and generate an out-of-distribution robustness score based on the following criteria: 
        The score should reflect the level of out-of-distribution robustness of the input text: 
        the higher the out-of-distribution robustness, the closer the score should be to 1; the lower the out-of-distribution robustness, the closer the score should be to 0. 

        If the text lacks terms and meanings related to out-of-distribution robustness or is unrelated to out-of-distribution robustness, return a score of 0.

        Additionally, provide a CONCISE reason that justifies this score based on the concept of OOD robustness.

        **
        IMPORTANT: Please ensure that the output is in JSON format, with the 'score' key providing the OOD robustness score and 
        the 'reason' key providing the rationale.
        Example JSON:
        {{
            "score": <ood_robustness_score>,
            "reason": "The score is <ood_robustness_score> because <your_reason>."
        }}

        If the score is 1, keep it short and say something positive with an encouraging tone 
        (but don't overdo it, otherwise, it gets annoying).
        **

        input:
        {input}

        JSON:
        """

    @staticmethod
    def generate_verdict(input: str) -> str:
        return f"""
        ** 
        Please determine the model's out-of-distribution robustness based on the following concepts:
        - If the input contains phrases that indicate robustness to unseen data (e.g., 'generalizes well', 'handles unseen cases effectively'), return 'yes'.
        - If the input contains phrases indicating vulnerability to unseen data (e.g., 'fails on new data', 'struggles with OOD samples'), return 'no'.

        Example:
        {{
            "verdict": "no",
            "reason": "The model struggles with out-of-distribution samples, indicating poor generalization."
        }},
        {{
            "verdict": "yes",
            "reason": "The model demonstrates robustness by generalizing effectively to unseen data."
        }}

        input:
        {input}

        JSON:
        """
