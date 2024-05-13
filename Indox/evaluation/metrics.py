import numpy as np
from bert_score import BERTScorer
from collections import defaultdict

from .unieval import DialogEvaluator


def convert_to_json(output_list, src_list=None, ref_list=None, context_list=None, scores=None, doc_id=None,
                    system_id=None):
    """
        Convert the data into the json format.

        output_list: a list of model output
        src_list: source input for different NLG tasks. For example, source document for summarization
                  and dialogue history for dialogue response generation
        ref_list: human-annotated groundtruth
        context_list: the context needed to evaluate several specific dimension. For example,
                      additional factual information when evaluating engagingness and groundedness in dialogues
        scores: human scores for evaluating the model output. They can be used to calculate the correlation
                between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
        doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
        system_id: the index of the generation system. It can be used to calculate system-level correlation.
    """
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
        json_data.append(cur)
    return json_data


class DilaougesScorer(DialogEvaluator):
    def evaluate(self, inputs):
        query, answer, context = inputs['query'], inputs['answer'], inputs['context']
        cands = [answer[0] for _ in range(len(context))]
        queries = [query[0] for _ in range(len(context))]
        K = len(context)

        scores = defaultdict(list)
        for i in range(K):
            # Prepare data for pre-trained evaluators

            data = convert_to_json(output_list=[cands[i]],
                                   src_list=[queries[i]], context_list=[context[i]])
            score = self.single_evaluate(data)
            [scores[key].append(item) for key, item in score[0].items()]

        # print("\n\nUni Eval Sores")
        # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in scores.items()]
        # scores['K'].append(K)
        return scores, K
