import json
from typing import List
from loguru import logger
import sys
from indoxJudge.metrics import (
    Faithfulness,
    Privacy,
    Misinformation,
    MachineEthics,
    StereotypeBias,
    Fairness,
    Harmfulness,
    AnswerRelevancy,
    KnowledgeRetention,
    Hallucination,
    Toxicity,
    Bias,
    BertScore,
    BLEU,
    ContextualRelevancy,
    GEval,
    METEOR,
    Gruen,
    ToxicityDiscriminative,
    OutOfDistributionRobustness,
    AdversarialRobustness,
    Privacy,
    RobustnessToAdversarialDemonstrations,
    SafetyToxicity,
    FactualConsistency,
)

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class Evaluator:
    """
    The Evaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, Toxicity, BertScore, BLEU, Rouge, and METEOR.
    """

    def __init__(self, model, metrics: List):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            model: The language model to be evaluated.
            metrics (List): A list of metric instances to evaluate the model.
        """
        self.model = model
        self.metrics = metrics
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.evaluation_score = 0
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
                    self.evaluation_score += score
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
                    self.evaluation_score += score

                    self.metrics_score["KnowledgeRetention"] = round(score, 2)
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results["Hallucination"] = {
                        "score": score,
                        "reason": metric.reason,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.evaluation_score += score

                    self.metrics_score["Hallucination"] = round(score, 2)
                elif isinstance(metric, Toxicity):
                    score = metric.measure()
                    results["Toxicity"] = {
                        "score": score,
                        "reason": metric.reason,
                        "opinions": metric.opinions,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.evaluation_score += score

                    self.metrics_score["Toxicity"] = round(score, 2)

                elif isinstance(metric, Bias):
                    score = metric.measure()
                    results["Bias"] = {
                        "score": score,
                        "reason": metric.reason,
                        "opinions": metric.opinions,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.evaluation_score += score
                    self.metrics_score["Bias"] = round(score, 2)

                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    results["BertScore"] = {
                        "precision": score["Precision"],
                        "recall": score["Recall"],
                        "f1_score": score["F1-score"],
                    }
                    # self.evaluation_score += score

                    # self.metrics_score["BertScore"] = score
                    self.metrics_score["precision"] = score["Precision"]
                    self.evaluation_score += score["Precision"]

                    self.metrics_score["recall"] = score["Recall"]
                    self.evaluation_score += score["Recall"]

                    self.metrics_score["f1_score"] = score["F1-score"]
                    self.evaluation_score += score["F1-score"]

                elif isinstance(metric, BLEU):
                    score = metric.measure()
                    results["BLEU"] = {"score": score}
                    self.evaluation_score += score
                    self.metrics_score["BLEU"] = round(score, 2)

                elif isinstance(metric, Gruen):
                    score = metric.measure()
                    results["Gruen"] = {"score": score[0]}
                    self.evaluation_score += score[0]
                    self.metrics_score["Gruen"] = score[0]
                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results["AnswerRelevancy"] = {
                        "score": score,
                        "reason": metric.reason,
                        "statements": metric.statements,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.evaluation_score += score
                    self.metrics_score["AnswerRelevancy"] = round(score, 2)
                elif isinstance(metric, ContextualRelevancy):
                    res = metric.measure()
                    results["ContextualRelevancy"] = {
                        "verdicts": res["verdicts"],
                        "reason": res["reason"],
                        "score": res["score"],
                    }
                    self.evaluation_score += res["score"]
                    self.metrics_score["ContextualRelevancy"] = round(res["score"], 2)

                elif isinstance(metric, GEval):
                    geval_result = metric.g_eval()
                    results["GEval"] = geval_result.replace("\n", " ")
                    geval_data = json.loads(results["GEval"])
                    score = int(geval_data["score"]) / 8
                    self.evaluation_score += score
                    self.metrics_score["GEval"] = round(score, 2)
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results["Hallucination"] = {
                        "score": score,
                        "reason": metric.reason,
                        "verdicts": [verdict.dict() for verdict in metric.verdicts],
                    }
                    self.evaluation_score += 1 - score
                    self.metrics_score["Hallucination"] = 1 - round(score, 2)
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
                    self.evaluation_score += score
                    self.metrics_score["KnowledgeRetention"] = round(score, 2)

                elif isinstance(metric, SafetyToxicity):
                    score = 1 - metric.calculate_toxicity_score()
                    reason = metric.get_reason()
                    results["SafetyToxicity"] = {
                        "score": score,
                        "reason": reason.reason,
                    }
                    self.metrics_score["SafetyToxicity"] = round(score, 2)

                elif isinstance(metric, METEOR):
                    score = metric.measure()
                    results["METEOR"] = {"score": score}
                    self.evaluation_score += score
                    self.metrics_score["METEOR"] = score
                elif isinstance(metric, Fairness):
                    score = metric.calculate_fairness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Fairness"] = round(score, 2)
                    results["Fairness"] = reason.reason

                elif isinstance(metric, Harmfulness):
                    score = metric.calculate_harmfulness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Harmfulness"] = round(score, 2)
                    results["Harmfulness"] = reason.reason

                elif isinstance(metric, Privacy):
                    score = metric.calculate_privacy_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Privacy"] = round(score, 2)
                    results["Privacy"] = reason.reason

                elif isinstance(metric, Misinformation):
                    score = metric.calculate_misinformation_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Misinformation"] = round(score, 2)
                    results["Misinformation"] = reason.reason

                elif isinstance(metric, MachineEthics):
                    score = metric.calculate_ethics_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["MachineEthics"] = round(score, 2)
                    results["MachineEthics"] = reason.reason

                elif isinstance(metric, ToxicityDiscriminative):
                    score = metric.measure()
                    results["ToxicityDiscriminative"] = {"score": score}
                    self.evaluation_score += score
                    self.metrics_score["ToxicityDiscriminative"] = round(score, 2)

                elif isinstance(metric, StereotypeBias):
                    score = metric.calculate_stereotype_bias_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["StereotypeBias"] = round(score, 2)
                    results["StereotypeBias"] = reason.reason

                elif isinstance(metric, OutOfDistributionRobustness):
                    score = metric.calculate_ood_robustness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["OutOfDistributionRobustness"] = round(score, 2)
                    results["OutOfDistributionRobustness"] = {
                        "score": score,
                        "reason": reason.reason,
                    }

                elif isinstance(metric, AdversarialRobustness):
                    score = metric.calculate_robustness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["AdversarialRobustness"] = round(score, 2)
                    results["AdversarialRobustness"] = {
                        "score": score,
                        "reason": reason.reason,
                    }

                elif isinstance(metric, Privacy):
                    score = metric.calculate_privacy_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Privacy"] = round(score, 2)
                    results["Privacy"] = {"score": score, "reason": reason.reason}

                elif isinstance(metric, RobustnessToAdversarialDemonstrations):
                    score = metric.calculate_adversarial_robustness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["RobustnessToAdversarialDemonstrations"] = round(
                        score, 2
                    )
                    results["RobustnessToAdversarialDemonstrations"] = {
                        "score": score,
                        "reason": reason.reason,
                    }
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
        self.results = results
        # return results

    def plot(self, mode="external"):
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict

        graph_input = create_model_dict(
            name="LLM Evaluator",
            metrics=self.metrics_score,
            score=self.evaluation_score / len(self.metrics),
        )
        visualizer = Visualization(data=graph_input, mode="llm")
        return visualizer.plot(mode=mode)
