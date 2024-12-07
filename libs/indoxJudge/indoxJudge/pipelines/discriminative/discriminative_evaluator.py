import json
import sys
from typing import Tuple, Dict, List

from loguru import logger

from indoxJudge.metrics import ToxicityDiscriminative

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


from loguru import logger


class DiscriminativeToxicityEvaluator:
    def __init__(self, model, dataset_path, num_columns: int = None):
        """
        Initializes the discriminative evaluator with the specified model, dataset path,
        and number of columns to check.
        """
        try:
            self.model = model
            self.metrics = [
                ToxicityDiscriminative(
                    dataset_path=dataset_path, num_columns=num_columns
                )
            ]
            print("\n" + "=" * 50)

            logger.info("Evaluator initialized with model and metrics.")
            self.set_model_for_metrics()
            self.evaluation_score = 0
            self.metrics_score = {}
            self.metrics_reasons = {}

        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise

        except Exception as e:
            logger.error(f"An error occurred during initialization: {e}")
            raise

    def set_model_for_metrics(self):
        """
        Sets the model for all metrics.
        """
        try:
            for metric in self.metrics:
                if hasattr(metric, "set_model"):
                    metric.set_model(self.model)
            logger.info("Model set for all metrics.")

        except Exception as e:
            logger.error(f"An error occurred while setting the model for metrics: {e}")
            raise

    def judge(self):
        """
        Evaluates the dataset using the provided metrics and returns the results.
        """
        results = {}

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            try:
                logger.info(f"Evaluating metric: {metric_name}")

                # For ToxicityDiscriminative
                if isinstance(metric, ToxicityDiscriminative):
                    score = metric.measure()
                    results["ToxicityDiscriminative"] = {"accuracy_score": score}
                    self.metrics_score["ToxicityDiscriminative"] = score

                logger.info(f"Completed evaluation for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")

        return results

    def _evaluation_score_discriminative(self):
        from skcriteria import mkdm
        from skcriteria.madm import simple

        evaluation_metrics = self.metrics_score.copy()
        if "evaluation_score" in evaluation_metrics:
            del evaluation_metrics["evaluation_score"]
        weights = {"ToxicityDiscriminative": 0.05}

        metric_values = list(evaluation_metrics.values())
        weight_values = list(weights.values())

        dm = mkdm(
            matrix=[metric_values],
            objectives=[max] * len(metric_values),
            weights=weight_values,
            criteria=list(evaluation_metrics.keys()),
        )

        saw = simple.WeightedSumModel()
        rank = saw.evaluate(dm)
        final_score_array = rank.e_["score"]

        return round(final_score_array.item(), 2)

    def plot(self, mode="external", interpreter=None):
        """
        Plots the evaluation results.

        Args:
            mode (str): The mode for plotting. Default is "external".
            interpreter (object): An LLM model used to interpret the results before plotting. If None, the plot is generated without interpretation.

        """
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict

        metrics = self.metrics_score.copy()
        del metrics["evaluation_score"]
        score = self.metrics_score["evaluation_score"]
        graph_input = create_model_dict(
            name="Safety Evaluator", metrics=metrics, score=score
        )

        if interpreter:
            interpret = interpreter.generate_interpretation(
                models_data=graph_input, mode="safety"
            )
            parsed_response = json.loads(interpret)

            bar_chart = parsed_response.get(
                "bar_chart", "Bar chart interpretation not found."
            )
            gauge_chart = parsed_response.get(
                "gauge_chart", "Gauge chart interpretation not found."
            )
            chart_interpretations = {
                "Bar Chart": bar_chart,
                "Gauge Chart": gauge_chart,
            }
            visualization = Visualization(
                data=graph_input,
                mode="safety",
                chart_interpretations=chart_interpretations,
            )

        else:
            visualization = Visualization(data=graph_input, mode="safety")

        return visualization.plot(mode=mode)

    def format_for_analyzer(self, name):
        from indoxJudge.utils import create_model_dict

        metrics = self.metrics_score.copy()
        del metrics["evaluation_score"]
        score = self.metrics_score["evaluation_score"]
        analyzer_input = create_model_dict(name=name, score=score, metrics=metrics)
        return analyzer_input
