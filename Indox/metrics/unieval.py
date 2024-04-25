import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sys
import numpy as np
from nltk import sent_tokenize


class UniEvaluator:
    def __init__(self, model_name_or_path, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up model """
        self.device = device
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config,
                                                           cache_dir=cache_dir)

        self.model.eval()
        self.model.to(device)

        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]

    def score(self, inputs, batch_size=8):
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in (range(0, len(inputs), batch_size)):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)

                    pos_score = self.softmax(logits)[:, self.pos_id]  # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id]  # No

                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    pos_score_list += cur_pos_score
                    neg_score_list += cur_neg_score

            except RuntimeError:
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)

        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))

        return score_list


class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-dialog',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.dimensions = ['naturalness', 'coherence', 'engagingness',
                           'groundedness', 'understandability']

    def single_evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, DialogEvaluator will evaluate
                  five dimensions: naturalness, coherence, engagingness, groundedness and understandability.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            # print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate summation score for 'engagingness'
            if dim == 'engagingness':
                src_list, output_list, context_list = [], [], []
                n_sents = []  # the number of sentences in each generated response
                for i in range(n_data):
                    source = data[i]['source']
                    context = data[i]['context']
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        context_list.append(context)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list)
                sent_score = self.scorer.score(input_list)

                # Get the summation score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]))
                    start_idx += cur_n_sent

            # Calculate turn-level score for other dimensions
            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list, output_list, context_list = [], [], []
                for i in range(n_data):
                    if dim == 'coherence':
                        src_list.append(data[i]['source'])
                    else:
                        src_list.append('')
                    output_list.append(data[i]['system_output'])
                    if dim == 'groundedness':
                        context_list.append(data[i]['context'])
                    else:
                        context_list.append('')
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list)
                score = self.scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        # if print_result == True:
        #     print_scores(eval_scores)

        return eval_scores

def add_question(dimension, output, src=None, ref=None, context=None):
    """
        Add questions to generate input in Bool-QA format for UniEval.

        dimension: specific dimension to be evaluated
        src: source input for different NLG tasks. For example, source document for summarization
             and dialogue history for dialogue response generation.
        output: output text generated by the models
        ref: human-annotataed groundtruth
        context: the context needed to evaluate several specific dimension. For example,
                 additional factual information when evaluating engagingness and groundedness in dialogues.
    """

    input_with_question = []
    for i in range(len(output)):

        if dimension == 'naturalness':
            cur_input = 'question: Is this a natural response in the dialogue? </s> response: ' + output[i]
        elif dimension == 'coherence':
            cur_input = 'question: Is this a coherent response given the dialogue history? </s> response: ' \
                        + output[i] + ' </s> dialogue history: ' + src[i]
        elif dimension == 'engagingness':
            cur_input = 'question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: ' \
                        + output[i] + ' </s> dialogue history: ' + src[i] + ' </s> fact: ' + context[i]
        elif dimension == 'groundedness':
            cur_input = 'question: Is this response consistent with knowledge in the fact? </s> response: ' \
                        + output[i] + ' </s> fact: ' + context[i]
        elif dimension == 'understandability':
            cur_input = 'question: Is this an understandable response in the dialogue? </s> response: ' + output[i]
        else:
            raise NotImplementedError(
                'The input format for this dimension is still undefined. Please customize it first.')

        input_with_question.append(cur_input)
    return input_with_question