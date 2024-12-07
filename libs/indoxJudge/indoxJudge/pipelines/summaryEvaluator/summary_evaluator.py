from typing import Dict, List, Union, Optional
from skcriteria import mkdm
from skcriteria.madm import simple
import numpy as np
from loguru import logger
from .evaluation_weights import EvaluationWeights
import sys
from indoxJudge.metrics import (
    StructureQuality,
    Conciseness,
    FactualConsistency,
    InformationCoverage,
    Relevance,
    SummaryGEval as GEval,
    SummaryRouge as Rouge,
    SummaryToxicity as Toxicity,
    SummaryBleu as Bleu,
    SummaryMeteor as Meteor,
    SummaryBertScore as BertScore,
)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class SummaryEvaluator:

    def __init__(self, llm_as_judge, source, summary):

        self.model = llm_as_judge
        self.metrics = [
            StructureQuality(summary=summary),
            Conciseness(summary=summary, source_text=source),
            FactualConsistency(summary=summary, source_text=source),
            InformationCoverage(summary=summary, source_text=source),
            Relevance(summary=summary, source_text=source),
            Rouge(generated_summary=summary, reference_summary=source),
            GEval(summary=summary),
            Toxicity(summary=summary),
            Bleu(summary=summary, source=source),
            Meteor(summary=summary, source=source),
            BertScore(generated=summary, reference=source),
        ]
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.metrics_score = {}
        self.results = {}
        self.weights = EvaluationWeights.get_all_weights()

    def set_model_for_metrics(self):
        """
        Sets the language model for each metric that requires it.
        """
        for metric in self.metrics:
            if hasattr(metric, "set_model"):
                metric.set_model(self.model)
        logger.info("Model set for all metrics.")

    def judge(self):
        """
        Evaluates the language model using the provided metrics and returns the results.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            try:
                print("\n" + "=" * 50)
                logger.info(f"Evaluating metric: {metric_name}")
                if isinstance(metric, StructureQuality):
                    res = metric.measure()
                    results["StructureQuality"] = {
                        "score": res["score"],
                        "verdicts": res["verdicts"],
                        "reason": res["reason"],
                    }
                    self.metrics_score["StructureQuality"] = res["score"]

                elif isinstance(metric, Conciseness):
                    res = metric.measure()
                    results["Conciseness"] = {
                        "score": res["overall_score"],
                        "metrics": res["metrics"],
                        "issues": res["issues"],
                        "suggestions": res["suggestions"],
                    }
                    self.metrics_score["Conciseness"] = res["overall_score"]
                elif isinstance(metric, FactualConsistency):
                    res = metric.measure()
                    results["FactualConsistency"] = {
                        "score": res["score"],
                        "summary_claims": res["summary_claims"],
                        "verified_claims": res["verified_claims"],
                        "category_scores": res["category_scores"],
                        "consistency_stats": res["consistency_stats"],
                    }
                    self.metrics_score["FactualConsistency"] = res["score"]
                elif isinstance(metric, InformationCoverage):
                    res = metric.measure()
                    results["InformationCoverage"] = {
                        "score": res["score"],
                        "information_elements": res["information_elements"],
                        "coverage_scores": res["coverage_scores"],
                        "coverage_stats": res["coverage_stats"],
                        "verdicts": res["verdicts"],
                    }

                    self.metrics_score["InformationCoverage"] = res["score"]
                elif isinstance(metric, Relevance):
                    res = metric.measure()
                    results["Relevance"] = {
                        "score": res["score"],
                        "key_points": res["key_points"],
                        "relevance_scores": res["relevance_scores"],
                        "key_point_coverage": res["key_point_coverage"],
                    }
                    self.metrics_score["Relevance"] = res["score"]

                elif isinstance(metric, Rouge):
                    res = metric.measure()
                    results["Rouge"] = {
                        "score": res["overall_score"],
                        "verdict": res["verdict"],
                        "detailed_scores": res["detailed_scores"],
                    }
                    self.metrics_score["Rouge"] = res["overall_score"]

                elif isinstance(metric, GEval):
                    res = metric.measure()
                    results["GEval"] = {
                        "score": res["score"],
                        "grammar_issues": res["grammar_issues"],
                        "grammar_scores": res["grammar_scores"],
                        "issue_distribution": res["issue_distribution"],
                    }
                    self.metrics_score["GEval"] = res["score"]

                elif isinstance(metric, Toxicity):
                    res = metric.measure()
                    results["Toxicity"] = {
                        "score": res["score"],
                        "toxic_elements": res["toxic_elements"],
                        "toxicity_scores": res["toxicity_scores"],
                        "element_distribution": res["element_distribution"],
                    }
                    self.metrics_score["Toxicity"] = res["score"]
                elif isinstance(metric, Bleu):
                    res = metric.measure()
                    results["Bleu"] = {
                        "score": res["overall_score"],
                        "dic": res["detailed_scores"],
                    }

                    self.metrics_score["Bleu"] = res["overall_score"]
                elif isinstance(metric, Meteor):
                    res = metric.measure()
                    results["Meteor"] = {
                        "score": res["overall_score"],
                        "verdict": res["verdict"],
                        "detailed_scores": res["detailed_scores"],
                    }
                    self.metrics_score["Meteor"] = res["overall_score"]
                elif isinstance(metric, BertScore):
                    res = metric.measure()
                    results["BertScore"] = {
                        "score": res["overall_score"],
                        "verdict": res["verdict"],
                        "detailed_scores": res["detailed_scores"],
                    }

                    self.metrics_score["BertScore"] = res["overall_score"]

                logger.info(
                    f"Completed evaluation for metric: {metric_name}, score: {self.metrics_score[metric_name]}"
                )

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")

        # Calculate the final evaluation score after all metrics have been processed
        # evaluation_score = self._evaluation_score_llm_mcda()

        # self.metrics_score["evaluation_score"] = evaluation_score

        # results["evaluation_score"] = evaluation_score
        self.results = results

    ## MCDA Score
    def _normalize_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0,1] range and handle inverse metrics"""
        normalized = metrics.copy()

        # Inverse metrics (where lower is better)
        inverse_metrics = {"Toxicity"}
        for metric in inverse_metrics:
            if metric in normalized:
                normalized[metric] = 1 - normalized[metric]

        # Ensure all values are in [0,1]
        for key, value in normalized.items():
            if value < 0:
                normalized[key] = 0
            elif value > 1:
                normalized[key] = 1

        return normalized

    def _validate_metrics_and_weights(self) -> None:
        """Validate that all metrics have corresponding weights and vice versa"""
        metric_set = set(self.metrics_score.keys()) - {"evaluation_score"}
        weight_set = set(self.weights.keys())

        missing_weights = metric_set - weight_set
        unused_weights = weight_set - metric_set

        if missing_weights:
            raise ValueError(f"Missing weights for metrics: {missing_weights}")
        if unused_weights:
            logger.warning(f"Unused weights for metrics: {unused_weights}")

    def calculate_evaluation_score(self) -> float:
        """Calculate final evaluation score using MCDA"""
        try:
            # Remove existing evaluation score if present
            evaluation_metrics = self.metrics_score.copy()
            evaluation_metrics.pop("evaluation_score", None)

            # Validate metrics and weights
            self._validate_metrics_and_weights()

            # Normalize scores
            normalized_metrics = self._normalize_scores(evaluation_metrics)

            # Prepare data for MCDA
            metric_names = list(normalized_metrics.keys())
            metric_values = [normalized_metrics[metric] for metric in metric_names]
            weight_values = [self.weights[metric] for metric in metric_names]

            # Create decision matrix
            dm = mkdm(
                matrix=[metric_values],
                objectives=[max] * len(metric_values),
                weights=weight_values,
                criteria=metric_names,
            )

            # Apply Weighted Sum Model
            saw = simple.WeightedSumModel()
            rank = saw.evaluate(dm)
            final_score = rank.e_["score"].item()

            # Ensure score is in [0,1] range and round to 2 decimals
            final_score = max(0, min(1, final_score))
            return round(final_score, 2)

        except Exception as e:
            logger.error(f"Error calculating evaluation score: {str(e)}")
            raise

    def get_metric_contributions(self) -> Dict[str, float]:
        """Calculate how much each metric contributes to final score"""
        normalized_metrics = self._normalize_scores(self.metrics_score)
        contributions = {}

        for metric, score in normalized_metrics.items():
            if metric != "evaluation_score":
                weight = self.weights.get(metric, 0)
                contributions[metric] = round(score * weight, 4)

        return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))

    # def _evaluation_score_llm_mcda(self):
    #     from skcriteria import mkdm
    #     from skcriteria.madm import simple

    #     evaluation_metrics = self.metrics_score.copy()
    #     if "evaluation_score" in evaluation_metrics:
    #         del evaluation_metrics["evaluation_score"]
    #     # Transform the values for Bias, Hallucination, and Toxicity
    #     evaluation_metrics["Bias"] = 1 - evaluation_metrics["Bias"]
    #     evaluation_metrics["Hallucination"] = 1 - evaluation_metrics["Hallucination"]
    #     evaluation_metrics["Toxicity"] = 1 - evaluation_metrics["Toxicity"]

    #     # Weights for each metric
    #     weights = {
    #         'StructureQuality': ,
    #         'Conciseness': ,
    #         'FactualConsistency': ,
    #         'InformationCoverage': ,
    #         'Relevance':,
    #         'Rouge': ,
    #         'GEval': ,
    #         'Toxicity': ,
    #         'Bleu': ,
    #         'Meteor': ,
    #         'BertScore':
    #     }

    #     # Convert metrics and weights to lists
    #     metric_values = list(evaluation_metrics.values())
    #     weight_values = list(weights.values())

    #     # Create decision matrix
    #     dm = mkdm(
    #         matrix=[metric_values],
    #         objectives=[max] * len(metric_values),
    #         weights=weight_values,
    #         criteria=list(evaluation_metrics.keys()),
    #     )

    #     # Additive Weighting (SAW) method
    #     saw = simple.WeightedSumModel()
    #     rank = saw.evaluate(dm)
    #     final_score_array = rank.e_["score"]

    #     return round(final_score_array.item(), 2)

    # def plot(self, mode="external", interpreter=None):
    #     """
    #     Plots the evaluation results.

    #     Args:
    #         mode (str): The mode for plotting. Default is "external".
    #         interpreter (object): An LLM model used to interpret the results before plotting. If None, the plot is generated without interpretation.

    #     """
    #     from indoxJudge.graph import Visualization
    #     from indoxJudge.utils import create_model_dict

    #     metrics = self.metrics_score.copy()
    #     del metrics["evaluation_score"]
    #     score = self.metrics_score["evaluation_score"]
    #     graph_input = create_model_dict(
    #         name="LLM Evaluator", metrics=metrics, score=score
    #     )

    #     if interpreter:
    #         interpret = interpreter.generate_interpretation(
    #             models_data=graph_input, mode="llm"
    #         )
    #         parsed_response = json.loads(interpret)

    #         bar_chart = parsed_response.get(
    #             "bar_chart", "Bar chart interpretation not found."
    #         )
    #         radar_chart = parsed_response.get(
    #             "radar_chart", "Radar chart interpretation not found."
    #         )
    #         line_plot = parsed_response.get(
    #             "line_plot", "Line plot interpretation not found."
    #         )
    #         gauge_chart = parsed_response.get(
    #             "gauge_chart", "Gauge chart interpretation not found."
    #         )
    #         scatter_plot = parsed_response.get(
    #             "scatter_plot", "Scatter plot interpretation not found."
    #         )
    #         heatmap = parsed_response.get(
    #             "heatmap", "Heatmap interpretation not found."
    #         )

    #         # Create a dictionary with the extracted interpretations
    #         chart_interpretations = {
    #             "Bar Chart": bar_chart,
    #             "Radar Chart": radar_chart,
    #             "Line Plot": line_plot,
    #             "Gauge Chart": gauge_chart,
    #             "Scatter Plot": scatter_plot,
    #             "Heatmap": heatmap,
    #         }
    #         visualization = Visualization(
    #             data=graph_input,
    #             mode="llm",
    #             chart_interpretations=chart_interpretations,
    #         )

    #     else:
    #         visualization = Visualization(data=graph_input, mode="llm")

    #     return visualization.plot(mode=mode)

    # def format_for_analyzer(self, name):
    #     from indoxJudge.utils import create_model_dict

    #     metrics = self.metrics_score.copy()
    #     del metrics["evaluation_score"]
    #     score = self.metrics_score["evaluation_score"]
    #     analyzer_input = create_model_dict(name=name, score=score, metrics=metrics)
    #     return analyzer_input
