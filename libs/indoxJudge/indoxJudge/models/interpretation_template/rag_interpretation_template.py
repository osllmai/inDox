class RAGEvaluationTemplate:
    @staticmethod
    def generate_interpret(data: str) -> str:
        return f"""In an RAG evaluation app, the following are the results of the RAG pipeline. Your task is to provide a JSON response that directly interprets the model performances based on the given chart types, without explaining how each chart works. Focus on evaluating the RAG pipeline, identifying patterns, and highlighting key differences for each chart type.

        **
        IMPORTANT: Only return the response in JSON format. Your analysis should focus on key differences and patterns for each chart type.
        Example JSON:
        {{
            "bar_chart": "In the RAG pipeline evaluation, Model_2 shows superior precision (0.75) and f1_score (0.78) compared to Model_1 and Model_3. Model_1 performs well in recall (0.82) but lacks balance in precision, leading to a lower f1_score (0.71). Model_3 has a consistent but average performance across metrics, with a recall of 0.76 and f1_score of 0.73. The chart suggests that Model_2 is the best-performing model overall, especially in classification metrics.",

            "gauge_chart": "The gauge chart highlights that Model_2 performs the best in Faithfulness, nearing the maximum value. Model_1 shows room for improvement in Faithfulness, scoring lower on the scale compared to other models. Model_3 is somewhere in between but still falls short of Model_2, which excels in both Hallucination and Faithfulness metrics."
        }}

        Results:
        {data}

        Please provide a JSON response with the following chart interpretations:

        - Bar Chart: Compare specific metrics (e.g., precision, recall, f1_score) between models, emphasizing which models excel or fall behind in each metric.
        - Gauge Chart: Evaluate individual model performance for a specific metric (e.g., Faithfulness), and indicate how close each model comes to the ideal value for that metric.
        **

        JSON:
        """
