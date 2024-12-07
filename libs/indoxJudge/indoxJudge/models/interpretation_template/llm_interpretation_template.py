class LLMEvaluatorTemplate:
    @staticmethod
    def generate_interpret(data: str) -> str:
        return f"""In an LLM evaluation app, the following are the results of the evaluation pipeline for Model. Your task is to provide a JSON response that directly interprets the model's performance based on the given chart types, without explaining how each chart works. Focus on evaluating the metrics, identifying patterns, and highlighting key insights for each chart type.

        **
        IMPORTANT: Only return the response in JSON format. Your analysis should focus on key insights and patterns for each chart type.
        Example JSON:
        {{
            "bar_chart": "The bar chart shows Model's performance across various metrics. The model excels in Faithfulness (0.92) and AnswerRelevancy (0.88), indicating strong capabilities in these crucial areas. However, it shows room for improvement in Hallucination prevention (0.35, where lower is better) and Toxicity reduction (0.25, where lower is better). The model's performance in BLEU score (0.72) is commendable, suggesting good translation or text generation quality.",

            "radar_chart": "The radar chart provides a holistic view of Model's performance profile. The model shows balanced performance across most metrics, with notable strengths in KnowledgeRetention and BertScore. There's a slight dip in the Bias axis, indicating an area for potential improvement. The even distribution across other axes suggests that the model has been well-calibrated for overall performance, with no significant blind spots.",

            "line_plot": "The line plot illustrates Model's performance trends across different aspects. There's a noticeable upward trend in Faithfulness and AnswerRelevancy scores as we move from basic to more complex evaluation scenarios. However, the Hallucination score shows a slight increase in more challenging contexts, suggesting that the model might struggle with maintaining consistent factual accuracy in complex situations.",

            "gauge_chart": "The gauge chart for Faithfulness shows Model performing exceptionally well, with the needle pointing close to the maximum value. This indicates that the model has been highly optimized for providing faithful and accurate responses. The Toxicity gauge, where lower is better, shows the needle in the green zone, confirming the model's ability to generate safe, non-toxic content in most scenarios.",

            "scatter_plot": "The scatter plot comparing Faithfulness and Hallucination reveals a strong negative correlation, with data points clustered in the high Faithfulness, low Hallucination region. This suggests that {model_name} consistently maintains factual accuracy while minimizing hallucinated outputs. There are a few outliers in the mid-range of both metrics, possibly indicating specific scenarios where the model faces challenges in balancing these two aspects.",

            "heatmap": "The heatmap reveals interesting correlations between metrics for Model. There's a strong positive correlation between Faithfulness and AnswerRelevancy, suggesting that improvements in one area positively impact the other. Conversely, there's a negative correlation between Hallucination and KnowledgeRetention, indicating that as the model retains knowledge better, its tendency to hallucinate decreases. The heatmap also highlights a moderate correlation between BLEU score and BertScore, which is expected given that both are measures of text quality."
        }}

        Results:
        {data}

        Please provide a JSON response with the following chart interpretations for Model:

        - Bar Chart: Analyze the model's performance across different metrics (Faithfulness, AnswerRelevancy, Bias, Gruen, KnowledgeRetention, BLEU, Hallucination, Toxicity, BertScore), highlighting strengths and areas for improvement.
        - Radar Chart: Provide a holistic view of the model's performance profile, noting balanced areas and any axes that stand out (positively or negatively).
        - Line Plot: Interpret any trends or patterns in the model's performance across different aspects or evaluation scenarios.
        - Gauge Chart: Evaluate the model's performance for specific metrics (e.g., Faithfulness, Toxicity), indicating how close it comes to ideal values.
        - Scatter Plot: Analyze the relationship between two metrics (e.g., Faithfulness vs Hallucination), noting any correlations or clusters.
        - Heatmap: Identify correlations between different metrics, highlighting strong positive or negative relationships.
        **

        JSON:
        """