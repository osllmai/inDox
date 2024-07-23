import numpy as np
import math
from collections import Counter
from typing import List, Union
from utils.preprocessing import TextPreprocessor


class BLEU:
    def __init__(self, n: int = 2, remove_repeating_ngrams: bool = False):
        """
        Initialize the BLEU evaluator with the desired n-gram size and option to remove repeating n-grams.

        Parameters:
        n (int): The maximum size of the n-grams to use for evaluation (default is 2).
        remove_repeating_ngrams (bool): Whether to remove repeating n-grams (default is False).
        """
        self.n = n
        self.remove_repeating_ngrams = remove_repeating_ngrams

    def preprocess_text(self, text: str) -> str:
        preprocessor = TextPreprocessor()
        preprocessing_methods = [
            preprocessor.to_lower,
            preprocessor.keep_alpha_numeric,
            preprocessor.remove_number,
            preprocessor.remove_stopword,
            preprocessor.lemmatize_word,
        ]
        preprocessed_text = preprocessor.preprocess_text(text, preprocessing_methods)
        return preprocessed_text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def get_ngrams(self, text: str, n: int) -> List[str]:
        tokens = self.tokenize(text)
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if self.remove_repeating_ngrams:
            ngrams = list(set(ngrams))
        return ngrams

    def calculate_bp(self, context_length: int, llm_answer_length: int) -> float:
        if llm_answer_length > context_length:
            return 1
        else:
            penalty = 1 - (context_length / llm_answer_length)
            return np.exp(penalty)

    def calculate_clipped_precision(
        self, context_ngrams: Counter, llm_answer_ngrams: Counter
    ) -> float:
        clipped_count = 0
        total_count = sum(llm_answer_ngrams.values())

        for ngram in llm_answer_ngrams:
            clipped_count += min(llm_answer_ngrams[ngram], context_ngrams.get(ngram, 0))

        return clipped_count / total_count if total_count > 0 else 0

    def calculate_bleu(self, context: str, llm_answer: str) -> float:
        context = self.preprocess_text(context)
        llm_answer = self.preprocess_text(llm_answer)

        context_length = len(self.tokenize(context))
        llm_answer_length = len(self.tokenize(llm_answer))

        BP = self.calculate_bp(context_length, llm_answer_length)

        clipped_precision_scores = []
        for i in range(1, self.n + 1):
            context_ngrams = Counter(self.get_ngrams(context, i))
            llm_answer_ngrams = Counter(self.get_ngrams(llm_answer, i))

            clipped_precision = self.calculate_clipped_precision(
                context_ngrams, llm_answer_ngrams
            )
            clipped_precision_scores.append(
                clipped_precision if clipped_precision > 0 else 1e-9
            )

        weights = [1 / self.n] * self.n
        s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_scores))
        score = BP * math.exp(math.fsum(s))

        return score

    def __call__(
        self, context: Union[str, List[str]], llm_answer: Union[str, List[str]]
    ) -> float:
        if isinstance(context, str):
            context = [context]

        if isinstance(llm_answer, list):
            llm_answer = " ".join(llm_answer)

        scores = []
        for ctx in context:
            scores.append(self.calculate_bleu(ctx, llm_answer))
        average_score = np.mean(scores)
        return average_score
