class ModelComparisonTemplate:
    @staticmethod
    def generate_comparison(models: str, mode: str) -> str:
        return f"""In an LLM evaluation app, the following are the results of different models in {mode} pipeline. Your task is to provide a JSON response that directly interprets the model performances based on the given chart types, without explaining how each chart works. Focus on comparing the models, identifying patterns, and highlighting key differences for each chart type.

        **
        IMPORTANT: Only return the response in JSON format. Your analysis should focus on key differences and patterns for each chart type.
        Example JSON:
        {{
            "radar_chart": "Based on the results, Model_2 outperforms the other models in Faithfulness, AnswerRelevancy, and KnowledgeRetention with perfect scores. Model_1 and Model_3 are similar in many metrics, but Model_1 has lower performance in Hallucination and KnowledgeRetention, while Model_3 shows higher Hallucination compared to Model_1. Overall, Model_2 shows the best balance across all metrics.",

            "bar_chart": "Model_2 has the highest precision (0.667) and f1_score (0.71), while Model_1 and Model_3 both have slightly lower f1_scores at 0.70. In recall, all models perform similarly, with Model_1 and Model_2 leading at 0.77. Model_2 clearly excels in key metrics such as precision and f1_score, indicating it is the best-performing model overall in terms of classification ability.",

            "scatter_plot": "When comparing precision and recall, all models cluster closely, indicating similar balance between these two metrics. Model_2 has a slight edge in precision, while Model_1 and Model_3 maintain similar positions, suggesting that there are no extreme trade-offs between precision and recall for any model.",

            "line_plot": "The line plot of BLEU scores shows a gradual improvement from Model_1 to Model_2, with Model_2 having the highest BLEU score (0.14). Model_3 has a slightly lower score than Model_2 but still outperforms Model_1. This trend suggests that Model_2 consistently improves in translation-related metrics.",

            "heatmap": "The heatmap reveals a negative correlation between Faithfulness and Hallucination across all models. As Faithfulness improves (especially in Model_2), Hallucination tends to decrease. There’s also a positive correlation between precision, f1_score, and recall, with Model_2 showing the strongest overall performance in these areas.",

            "gauge_chart": "Model_2 achieves a perfect score in Faithfulness, with its gauge close to the maximum, indicating superior performance. Model_1, on the other hand, has lower Faithfulness and Hallucination scores, suggesting more room for improvement in these areas.",

            "table": "When looking at the overall results in the table plot, Model_2 consistently outperforms the other models across most metrics. Model_1 struggles in Faithfulness and KnowledgeRetention but performs adequately in precision and recall. Model_3 is slightly more balanced but suffers from higher Hallucination. Overall, Model_2 emerges as the most reliable model across all metrics, making it the top-performing model based on the aggregated results from the table plot."
        }}

        Results:
        {models}

        Please provide a JSON response with the following chart interpretations:

        - Radar Chart: Interpret the model performances across multiple metrics, focusing on the strengths and weaknesses of each model in comparison.
        - Bar Chart: Compare specific metrics (e.g., precision, recall, f1_score) between models, emphasizing which models excel or fall behind in each metric.
        - Scatter Plot: Identify relationships or trade-offs between two specific metrics (e.g., precision vs recall) across the models, and note any patterns or trends.
        - Line Plot: Analyze trends in specific metrics (e.g., score, BLEU) across the models, highlighting performance changes or consistencies.
        - Heatmap: Point out correlations or patterns between different metrics (e.g., Faithfulness and Hallucination) across models, with focus on any strong positive or negative correlations.
        - Gauge Chart: Evaluate individual model performance for a specific metric (e.g., Faithfulness), and indicate how close each model comes to the ideal value for that metric.
        - Table: Provide an overall conclusion based on all the metrics, summarizing which model performs the best across the board.
        **

        JSON:
        """
