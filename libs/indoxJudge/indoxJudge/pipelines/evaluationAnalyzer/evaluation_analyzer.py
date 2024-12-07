import json


class EvaluationAnalyzer:
    """
    This class is responsible for analyzing evaluation metrics for different models and generating plots based on their scores and performance.

    Attributes:
        models (list): A list of dictionaries containing model information, including name, score, and various evaluation metrics.

    Example:
        models = [
            {'name': 'Model_1',
             'score': 0.50,
             'metrics': {'Faithfulness': 0.55,
                         'AnswerRelevancy': 1.0,
                         'Bias': 0.45,
                         'Hallucination': 0.8,
                         'KnowledgeRetention': 0.0,
                         'Toxicity': 0.0,
                         'precision': 0.64,
                         'recall': 0.77,
                         'f1_score': 0.70,
                         'BLEU': 0.11}},
            {'name': 'Model_2',
             'score': 0.61,
             'metrics': {'Faithfulness': 1.0,
                         'AnswerRelevancy': 1.0,
                         'Bias': 0.0,
                         'Hallucination': 0.8,
                         'KnowledgeRetention': 1.0,
                         'Toxicity': 0.0,
                         'precision': 0.667,
                         'recall': 0.77,
                         'f1_score': 0.71,
                         'BLEU': 0.14}},
            {'name': 'Model_3',
             'score': 0.050,
             'metrics': {'Faithfulness': 1.0,
                         'AnswerRelevancy': 1.0,
                         'Bias': 0.0,
                         'Hallucination': 0.83,
                         'KnowledgeRetention': 0.0,
                         'Toxicity': 0.0,
                         'precision': 0.64,
                         'recall': 0.76,
                         'f1_score': 0.70,
                         'BLEU': 0.10}}
        ]

    Methods:
        plot(mode="external", interpreter=None):
            Generates plots for model evaluation. If `interpreter` is provided, it uses an LLM model for interpreting the results before plotting.
    """

    def __init__(self, models):
        self.models = models

    def plot(self, mode="external", interpreter=None):
        """
        Generates plots for model evaluation based on the provided models and metrics.

        Args:
            mode (str): Determines the plotting mode (default is "external").
            interpreter (object): An LLM model used to interpret the results before plotting. If None, the plot is generated without interpretation.

        Returns:
            Visualization: A plot representing the models' metrics and evaluation results.
        """
        from indoxJudge.graph import Visualization
        if interpreter:
            interpret = interpreter.generate_interpretation(models_data=self.models, mode="comparison")
            parsed_response = json.loads(interpret)

            # Extract interpretations for each chart type
            radar_chart = parsed_response.get('radar_chart', 'Radar chart interpretation not found.')
            bar_chart = parsed_response.get('bar_chart', 'Bar chart interpretation not found.')
            scatter_plot = parsed_response.get('scatter_plot', 'Scatter plot interpretation not found.')
            line_plot = parsed_response.get('line_plot', 'Line plot interpretation not found.')
            heatmap = parsed_response.get('heatmap', 'Heatmap interpretation not found.')
            gauge_chart = parsed_response.get('gauge_chart', 'Gauge chart interpretation not found.')
            table = parsed_response.get('table', 'table interpretation not found.')
            # Create a dictionary with the extracted interpretations
            chart_interpretations = {
                'Radar Chart': radar_chart,
                'Bar Chart': bar_chart,
                'Scatter Plot': scatter_plot,
                'Line Plot': line_plot,
                'Heatmap': heatmap,
                'Gauge Chart': gauge_chart,
                'Table': table
            }
            visualization = Visualization(data=self.models, mode="llm", chart_interpretations=chart_interpretations)

        else:
            visualization = Visualization(data=self.models, mode="llm")

        return visualization.plot(mode=mode)
