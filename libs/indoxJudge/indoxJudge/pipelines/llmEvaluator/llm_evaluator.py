import json

from loguru import logger
import sys
from indoxJudge.metrics import (
    Faithfulness,
    AnswerRelevancy,
    Bias,
    Gruen,
    KnowledgeRetention,
    BLEU,
    Hallucination,
    Toxicity,
    BertScore,
)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class LLMEvaluator:
    """
    The Evaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, Toxicity, BertScore, BLEU, and Gruen.
    """

    def __init__(self, llm_as_judge, llm_response, retrieval_context, query):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            llm_as_judge: The language model .
        """
        self.model = llm_as_judge
        if isinstance(retrieval_context, list):
            retrieval_context_join = "\n".join(retrieval_context)
        else:
            retrieval_context_join = retrieval_context
        self.metrics = [
            Faithfulness(
                llm_response=llm_response, retrieval_context=retrieval_context_join
            ),
            AnswerRelevancy(query=query, llm_response=llm_response),
            Bias(llm_response=llm_response),
            Hallucination(
                llm_response=llm_response, retrieval_context=retrieval_context_join
            ),
            KnowledgeRetention(
                messages=[{"query": query, "llm_response": llm_response}]
            ),
            Toxicity(messages=llm_response),
            BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            BLEU(llm_response=llm_response, retrieval_context=retrieval_context),
            Gruen(candidates=llm_response),
        ]
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.metrics_score = {}
        self.results = {}

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
                if isinstance(metric, Faithfulness):
                    res = metric.calculate_faithfulness_score()
                    results["Faithfulness"] = {
                        "claims": res["claims"],
                        "truths": res["truths"],
                        "verdicts": [verdict.__dict__ for verdict in res["verdicts"]],
                        "score": res["score"],
                        "reason": res["reason"],
                    }
                    score = res["score"]
                    self.metrics_score["Faithfulness"] = round(score, 2)

                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results["AnswerRelevancy"] = {
                        "score": score,
                        "reason": metric.reason,
                        "statements": metric.statements,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.metrics_score["AnswerRelevancy"] = round(score, 2)

                elif isinstance(metric, KnowledgeRetention):
                    score = metric.measure()
                    results["KnowledgeRetention"] = {
                        "score": score,
                        "reason": metric.reason,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                        "knowledges": [
                            knowledge.data for knowledge in metric.knowledges
                        ],
                    }
                    self.metrics_score["KnowledgeRetention"] = score
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results["Hallucination"] = {
                        "score": score,
                        "reason": metric.reason,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.metrics_score["Hallucination"] = round(score, 2)
                elif isinstance(metric, Toxicity):
                    score = metric.measure()
                    results["Toxicity"] = {
                        "score": score,
                        "reason": metric.reason,
                        "opinions": metric.opinions,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.metrics_score["Toxicity"] = round(score, 2)

                elif isinstance(metric, Bias):
                    score = metric.measure()
                    results["Bias"] = {
                        "score": score,
                        "reason": metric.reason,
                        "opinions": metric.opinions,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.metrics_score["Bias"] = round(score, 2)

                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    results["BertScore"] = {
                        "precision": score["Precision"],
                        "recall": score["Recall"],
                        "f1_score": score["F1-score"],
                    }
                    # self.metrics_score["BertScore"] = score
                    self.metrics_score["precision"] = score["Precision"]
                    self.metrics_score["recall"] = score["Recall"]
                    self.metrics_score["f1_score"] = score["F1-score"]

                elif isinstance(metric, BLEU):
                    score = metric.measure()
                    results["BLEU"] = {"score": score}
                    self.metrics_score["BLEU"] = score

                elif isinstance(metric, Gruen):
                    score = metric.measure()
                    results["Gruen"] = {"score": score[0]}
                    self.metrics_score["Gruen"] = score[0]
                if metric_name != "BertScore":
                    logger.info(
                        f"Completed evaluation for metric: {metric_name}, score: {self.metrics_score[metric_name]}"
                    )
                else:
                    logger.info(
                        f"""
Completed evaluation for metric: {metric_name}, scores: 
precision: {self.metrics_score["precision"]},
recall: {self.metrics_score["recall"]},
f1_score: {self.metrics_score["f1_score"]},
                        """
                    )

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")

        # Calculate the final evaluation score after all metrics have been processed
        evaluation_score = self._evaluation_score_llm_mcda()

        self.metrics_score["evaluation_score"] = evaluation_score

        results["evaluation_score"] = evaluation_score
        self.results = results
        # return results

    def _evaluation_score_llm_mcda(self):
        from skcriteria import mkdm
        from skcriteria.madm import simple

        evaluation_metrics = self.metrics_score.copy()
        if "evaluation_score" in evaluation_metrics:
            del evaluation_metrics["evaluation_score"]
        # Transform the values for Bias, Hallucination, and Toxicity
        evaluation_metrics["Bias"] = 1 - evaluation_metrics["Bias"]
        evaluation_metrics["Hallucination"] = 1 - evaluation_metrics["Hallucination"]
        evaluation_metrics["Toxicity"] = 1 - evaluation_metrics["Toxicity"]

        # Weights for each metric
        weights = {
            "Faithfulness": 0.2,
            "AnswerRelevancy": 0.15,
            "Bias": 0.1,
            "Hallucination": 0.15,
            "KnowledgeRetention": 0.1,
            "Toxicity": 0.1,
            "precision": 0.05,
            "recall": 0.05,
            "f1_score": 0.05,
            "BLEU": 0.025,
            "gruen": 0.025,
        }

        # Convert metrics and weights to lists
        metric_values = list(evaluation_metrics.values())
        weight_values = list(weights.values())

        # Create decision matrix
        dm = mkdm(
            matrix=[metric_values],
            objectives=[max] * len(metric_values),
            weights=weight_values,
            criteria=list(evaluation_metrics.keys()),
        )

        # Additive Weighting (SAW) method
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
            name="LLM Evaluator", metrics=metrics, score=score
        )

        if interpreter:
            interpret = interpreter.generate_interpretation(
                models_data=graph_input, mode="llm"
            )
            parsed_response = json.loads(interpret)

            bar_chart = parsed_response.get(
                "bar_chart", "Bar chart interpretation not found."
            )
            radar_chart = parsed_response.get(
                "radar_chart", "Radar chart interpretation not found."
            )
            line_plot = parsed_response.get(
                "line_plot", "Line plot interpretation not found."
            )
            gauge_chart = parsed_response.get(
                "gauge_chart", "Gauge chart interpretation not found."
            )
            scatter_plot = parsed_response.get(
                "scatter_plot", "Scatter plot interpretation not found."
            )
            heatmap = parsed_response.get(
                "heatmap", "Heatmap interpretation not found."
            )

            # Create a dictionary with the extracted interpretations
            chart_interpretations = {
                "Bar Chart": bar_chart,
                "Radar Chart": radar_chart,
                "Line Plot": line_plot,
                "Gauge Chart": gauge_chart,
                "Scatter Plot": scatter_plot,
                "Heatmap": heatmap,
            }
            visualization = Visualization(
                data=graph_input,
                mode="llm",
                chart_interpretations=chart_interpretations,
            )

        else:
            visualization = Visualization(data=graph_input, mode="llm")

        return visualization.plot(mode=mode)

    def format_for_analyzer(self, name):
        from indoxJudge.utils import create_model_dict

        metrics = self.metrics_score.copy()
        del metrics["evaluation_score"]
        score = self.metrics_score["evaluation_score"]
        analyzer_input = create_model_dict(name=name, score=score, metrics=metrics)
        return analyzer_input
