from collections import Counter
from typing import Union, List, Tuple, Dict
import numpy as np


class Rouge:
    """
    A class for computing ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.

    This class implements ROUGE-1, ROUGE-2, and ROUGE-L metrics to evaluate the quality of
    generated text (e.g., summaries, translations) against one or more reference texts.

    ROUGE-1 and ROUGE-2 measure the overlap of unigrams and bigrams respectively between
    the generated text and the reference text(s). ROUGE-L measures the longest common
    subsequence between the generated text and the reference text(s).

    Attributes:
        llm_response (str): The generated text to be evaluated.
        retrieval_context (Union[str, List[str]]): The reference text(s) to compare against.
    """

    def __init__(self, llm_response: str, retrieval_context: Union[str, List[str]]):
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context

    def measure(self) -> Dict[str, float]:
        """
        Compute the ROUGE scores between the actual response and the expected response(s).

        Returns:
            Dict[str, float]: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
                              Each score is a float rounded to three decimal places.
        """
        scores = self._calculate_scores(
            llm_answer=self.llm_response, context=self.retrieval_context
        )
        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text using various NLP techniques.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        from indoxJudge.utils import TextPreprocessor
        from indoxJudge.utils import nltk_download
        nltk_download()
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
        """
        Tokenize the input text into words.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        return text.split()

    def get_ngrams(self, text: str, n: int) -> List[Tuple[str]]:
        """
        Generate n-grams from the input text.

        Args:
            text (str): The input text to generate n-grams from.
            n (int): The size of the n-grams to generate.

        Returns:
            List[Tuple[str]]: A list of n-grams, where each n-gram is a tuple of strings.
        """
        tokens = self.tokenize(text)
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def count_matches(
        self, llm_answer_ngrams: List[Tuple[str]], context_ngrams: List[Tuple[str]]
    ) -> int:
        """
        Count the number of matching n-grams between two texts.

        Args:
            llm_answer_ngrams (List[Tuple[str]]): N-grams from the generated text.
            context_ngrams (List[Tuple[str]]): N-grams from the reference text.

        Returns:
            int: The number of matching n-grams.
        """
        llm_answer_counts = Counter(llm_answer_ngrams)
        context_counts = Counter(context_ngrams)
        matches = sum(
            min(context_counts[gram], llm_answer_counts[gram])
            for gram in context_counts
        )
        return matches

    def lcs(self, X, Y):
        """
        Compute the length of the Longest Common Subsequence (LCS) between two sequences.

        Args:
            X (List): The first sequence.
            Y (List): The second sequence.

        Returns:
            int: The length of the LCS.
        """
        m = len(X)
        n = len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        return L[m][n]

    def _calculate_scores(
        self, llm_answer: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.

        Args:
            llm_answer (Union[str, List[str]]): The generated text to evaluate.
            context (Union[str, List[str]]): The reference text(s) to compare against.

        Returns:
            Dict[str, float]: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        """
        if isinstance(llm_answer, list):
            llm_answer = " ".join(llm_answer)

        if isinstance(context, str):
            context = [context]

        llm_answer = self.preprocess_text(llm_answer)
        processed_contexts = [self.preprocess_text(ctx) for ctx in context]

        llm_answer_tokens = self.tokenize(llm_answer)

        rouge1_scores = []
        rouge2_scores = []
        rouge_l_scores = []

        for ctx in processed_contexts:
            context_tokens = self.tokenize(ctx)

            # ROUGE-1
            llm_answer_unigrams = self.get_ngrams(llm_answer, 1)
            context_unigrams = self.get_ngrams(ctx, 1)
            unigram_matches = self.count_matches(llm_answer_unigrams, context_unigrams)
            rouge1 = self._calculate_f1(
                unigram_matches, len(llm_answer_unigrams), len(context_unigrams)
            )
            rouge1_scores.append(rouge1)

            # ROUGE-2
            llm_answer_bigrams = self.get_ngrams(llm_answer, 2)
            context_bigrams = self.get_ngrams(ctx, 2)
            bigram_matches = self.count_matches(llm_answer_bigrams, context_bigrams)
            rouge2 = self._calculate_f1(
                bigram_matches, len(llm_answer_bigrams), len(context_bigrams)
            )
            rouge2_scores.append(rouge2)

            # ROUGE-L
            lcs_length = self.lcs(llm_answer_tokens, context_tokens)
            rouge_l = self._calculate_f1(
                lcs_length, len(llm_answer_tokens), len(context_tokens)
            )
            rouge_l_scores.append(rouge_l)

        scores = {
            "rouge1": round(np.mean(rouge1_scores), 3),
            "rouge2": round(np.mean(rouge2_scores), 3),
            "rougeL": round(np.mean(rouge_l_scores), 3),
        }

        return scores

    def _calculate_f1(self, matches: int, llm_length: int, ref_length: int) -> float:
        """
        Calculate the F1 score given the number of matches and the lengths of the compared texts.

        Args:
            matches (int): The number of matching units (e.g., n-grams, LCS length).
            llm_length (int): The length of the generated text in the relevant units.
            ref_length (int): The length of the reference text in the relevant units.

        Returns:
            float: The calculated F1 score.
        """
        precision = matches / llm_length if llm_length > 0 else 0
        recall = matches / ref_length if ref_length > 0 else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return f1
