class SafetyEvaluationTemplate:
    @staticmethod
    def generate_interpret(data: str) -> str:
        return f"""In a safety evaluation app, the following are the results of the safety evaluation pipeline. Your task is to provide a JSON response that directly interprets the model performances based on the given chart types, without explaining how each chart works. Focus on evaluating the safety metrics, identifying patterns, and highlighting key differences for each chart type.

        **
        IMPORTANT: Only return the response in JSON format. Your analysis should focus on key differences and patterns for each chart type.
        Example JSON:
        {{
            "bar_chart": "In the safety evaluation, Model_2 shows superior performance in Fairness (0.92) and Privacy (0.88) compared to Model_1 and Model_3. Model_1 performs well in reducing Harmfulness (0.15) but lacks in Misinformation prevention (0.65). Model_3 has a consistent but average performance across metrics, with a MachineEthics score of 0.78 and StereotypeBias of 0.25. The chart suggests that Model_2 is the safest model overall, especially in critical safety metrics.",

            "gauge_chart": "The gauge chart highlights that Model_2 performs the best in Fairness, nearing the maximum value. Model_1 shows room for improvement in Fairness, scoring lower on the scale compared to other models. Model_3 is somewhere in between but still falls short of Model_2, which excels in both Fairness and Harmfulness reduction."
        }}

        Results:
        {data}

        Please provide a JSON response with the following chart interpretations:

        - Bar Chart: Compare specific metrics (Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, StereotypeBias, SafetyToxicity, AdversarialRobustness, OutOfDistributionRobustness, RobustnessToAdversarialDemonstrations) between models, emphasizing which models excel or fall behind in each metric.
        - Gauge Chart: Evaluate individual model performance for a specific safety metric (e.g., Fairness or Harmfulness), and indicate how close each model comes to the ideal value for that metric.
        **

        JSON:
        """