from typing import List

from indox.IndoxEval.faithfulness.faithfulness import Faithfulness
from indox.IndoxEval.answer_relevancy.answerRelevancy import AnswerRelevancy
from indox.IndoxEval.bias.bias import Bias
from indox.IndoxEval.contextual_relevancy.contextualRelevancy import ContextualRelevancy


class Evaluator:
    def __init__(self, model, metrics: List):
        self.model = model
        self.metrics = metrics
        self.set_model_for_metrics()

    def set_model_for_metrics(self):
        for metric in self.metrics:
            if hasattr(metric, 'set_model'):
                metric.set_model(self.model)

    def evaluate(self):
        results = {}
        for metric in self.metrics:
            if isinstance(metric, Faithfulness):
                claims = metric.evaluate_claims()
                truths = metric.evaluate_truths()
                verdicts = metric.evaluate_verdicts(claims.claims)
                reason = metric.evaluate_reason(verdicts, truths.truths)
                results['faithfulness'] = {
                    'claims': claims.claims,
                    'truths': truths.truths,
                    'verdicts': [verdict.__dict__ for verdict in verdicts.verdicts],
                    'reason': reason.reason
                }
            elif isinstance(metric, AnswerRelevancy):
                score = metric.measure()
                results['answer_relevancy'] = {
                    'score': score,
                    'reason': metric.reason,
                    'statements': metric.statements,
                    'verdicts': [verdict.dict() for verdict in metric.verdicts]
                }
            elif isinstance(metric, Bias):
                score = metric.measure()
                results['bias'] = {
                    'score': score,
                    'reason': metric.reason,
                    'opinions': metric.opinions,
                    'verdicts': [verdict.dict() for verdict in metric.verdicts]
                }
            elif isinstance(metric, ContextualRelevancy):
                irrelevancies = metric.get_irrelevancies(metric.query, metric.retrieval_contexts)
                metric.set_irrelevancies(irrelevancies)
                verdicts = metric.get_verdicts(metric.query, metric.retrieval_contexts)
                reason = metric.get_reason(irrelevancies, metric.score)
                results['contextual_relevancy'] = {
                    'verdicts': [verdict.dict() for verdict in verdicts.verdicts],
                    'reason': reason.dict()
                }

            return results
