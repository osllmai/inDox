from typing import List, Dict, Set, Tuple
from pydantic import BaseModel, Field
from loguru import logger
import json
import re
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from .rougeTemplate import RougeTemplate
import nltk
import itertools

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelConfig:
    arbitrary_types_allowed = True


class RougeScore(BaseModel):
    precision: float
    recall: float
    f1_score: float

    class Config(ModelConfig):
        pass


class RougeVerdict(BaseModel):
    metric: str  # e.g., "rouge_1", "rouge_2", "rouge_l", "rouge_s"
    score: RougeScore
    details: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None

    class Config(ModelConfig):
        pass


class RougeScores(BaseModel):
    scores: List[RougeVerdict]

    class Config(ModelConfig):
        pass


class Rouge:
    def __init__(
        self,
        generated_summary: str,
        reference_summary: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
        skip_window: int = 4,
    ):
        self.generated = generated_summary
        self.reference = reference_summary
        self.include_reason = include_reason
        self.weights = weights or {
            "rouge_1": 0.6,
            "rouge_2": 0.25,
            "rouge_l": 0.1,
            "rouge_s": 0.05,
        }
        self.skip_window = skip_window
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        """Calculate ROUGE scores and return weighted average."""
        nltk.download("punkt")
        nltk.download("punkt_tab")
        self.rouge_scores = self._calculate_rouge_scores()
        self.score = self._calculate_weighted_score()
        if self.include_reason:
            self.verdict = self._generate_final_verdict()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return self._get_detailed_scores()

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ROUGE calculation."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep apostrophes
        text = re.sub(r"[^a-z0-9\'\s]", "", text)
        text = " ".join(text.split())
        return text

    def _get_skip_bigrams(self, tokens: List[str], max_gap: int) -> Counter:
        """Get skip-bigram counts with maximum gap size."""
        skip_bigrams = []
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + max_gap + 2, len(tokens))):
                skip_bigrams.append((tokens[i], tokens[j]))
        return Counter(skip_bigrams)

    def _get_ngrams(self, text: str, n: int) -> Counter:
        """Get n-gram counts from text."""
        tokens = word_tokenize(text)
        return Counter(tuple(gram) for gram in ngrams(tokens, n))

    def _calculate_rouge_s(self) -> Tuple[float, float, float, Dict]:
        """Calculate ROUGE-S (skip-bigram) scores with smoothing."""
        generated_clean = self._preprocess_text(self.generated)
        reference_clean = self._preprocess_text(self.reference)

        generated_tokens = word_tokenize(generated_clean)
        reference_tokens = word_tokenize(reference_clean)

        generated_skip_bigrams = self._get_skip_bigrams(
            generated_tokens, self.skip_window
        )
        reference_skip_bigrams = self._get_skip_bigrams(
            reference_tokens, self.skip_window
        )

        # Calculate overlap with smoothing
        overlap_bigrams = generated_skip_bigrams & reference_skip_bigrams
        overlap_count = sum(overlap_bigrams.values())

        # Add smoothing similar to BLEU
        smoothing = 1.0
        denominator_gen = sum(generated_skip_bigrams.values()) + smoothing
        denominator_ref = sum(reference_skip_bigrams.values()) + smoothing

        # Calculate scores with length penalty similar to BLEU
        precision = (overlap_count + smoothing) / denominator_gen
        recall = (overlap_count + smoothing) / denominator_ref

        # Length penalty factor (similar to BLEU's brevity penalty)
        beta = 1.0
        if len(generated_tokens) < len(reference_tokens):
            beta = np.exp(1 - len(reference_tokens) / len(generated_tokens))

        # Calculate F1 score with length penalty
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall) * beta
        else:
            f1 = 0

        details = {
            "matching_skip_bigrams": overlap_count,
            "total_generated_skip_bigrams": sum(generated_skip_bigrams.values()),
            "total_reference_skip_bigrams": sum(reference_skip_bigrams.values()),
            "length_penalty": beta,
        }

        return precision, recall, f1, details

    def _calculate_rouge_n(self, n: int) -> Tuple[float, float, float, Dict]:
        """Calculate ROUGE-N scores with smoothing."""
        generated_clean = self._preprocess_text(self.generated)
        reference_clean = self._preprocess_text(self.reference)

        generated_ngrams = self._get_ngrams(generated_clean, n)
        reference_ngrams = self._get_ngrams(reference_clean, n)

        # Calculate overlap with smoothing
        overlap_ngrams = generated_ngrams & reference_ngrams
        overlap_count = sum(overlap_ngrams.values())

        # Add smoothing similar to BLEU
        smoothing = 1.0 if n <= 2 else (0.5 ** (n - 2))
        denominator_gen = sum(generated_ngrams.values()) + smoothing
        denominator_ref = sum(reference_ngrams.values()) + smoothing

        precision = (overlap_count + smoothing) / denominator_gen
        recall = (overlap_count + smoothing) / denominator_ref

        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        details = {
            f"matching_{n}grams": overlap_count,
            f"total_generated_{n}grams": sum(generated_ngrams.values()),
            f"total_reference_{n}grams": sum(reference_ngrams.values()),
        }

        return precision, recall, f1, details

    def _calculate_rouge_l(self) -> Tuple[float, float, float, Dict]:
        """Calculate ROUGE-L scores with improved scoring."""
        generated_clean = self._preprocess_text(self.generated)
        reference_clean = self._preprocess_text(self.reference)

        generated_sents = sent_tokenize(generated_clean)
        reference_sents = sent_tokenize(reference_clean)

        generated_words = [word_tokenize(sent) for sent in generated_sents]
        reference_words = [
            word_tokenize(sent) for sent in reference_sents
        ]  # This line had the error

        # Calculate LCS with sentence-level aggregation
        lcs_lengths = []
        for ref_sent in reference_words:
            for gen_sent in generated_words:
                lcs_lengths.append(self._calculate_lcs(gen_sent, ref_sent))

        lcs_sum = sum(lcs_lengths)
        total_gen_words = sum(len(sent) for sent in generated_words)
        total_ref_words = sum(len(sent) for sent in reference_words)

        # Add smoothing
        smoothing = 1.0
        precision = (lcs_sum + smoothing) / (total_gen_words + smoothing)
        recall = (lcs_sum + smoothing) / (total_ref_words + smoothing)

        # Length penalty similar to BLEU
        beta = 1.0
        if total_gen_words < total_ref_words:
            beta = np.exp(1 - total_ref_words / total_gen_words)

        # Calculate F1 score with length penalty
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall) * beta
        else:
            f1 = 0

        details = {
            "lcs_sum": lcs_sum,
            "total_generated_words": total_gen_words,
            "total_reference_words": total_ref_words,
            "length_penalty": beta,
        }

        return precision, recall, f1, details

    def _calculate_lcs(self, s1: List[str], s2: List[str]) -> int:
        """Calculate Longest Common Subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _calculate_rouge_scores(self) -> List[RougeVerdict]:
        """Calculate all ROUGE scores."""
        scores = []

        # Calculate ROUGE-1
        p1, r1, f1, details1 = self._calculate_rouge_n(1)
        scores.append(
            RougeVerdict(
                metric="rouge_1",
                score=RougeScore(precision=p1, recall=r1, f1_score=f1),
                details=details1,
                reason="Unigram overlap assessment",
            )
        )

        # Calculate ROUGE-2
        p2, r2, f2, details2 = self._calculate_rouge_n(2)
        scores.append(
            RougeVerdict(
                metric="rouge_2",
                score=RougeScore(precision=p2, recall=r2, f1_score=f2),
                details=details2,
                reason="Bigram overlap assessment",
            )
        )

        # Calculate ROUGE-L
        pl, rl, fl, detailsl = self._calculate_rouge_l()
        scores.append(
            RougeVerdict(
                metric="rouge_l",
                score=RougeScore(precision=pl, recall=rl, f1_score=fl),
                details=detailsl,
                reason="Longest common subsequence assessment",
            )
        )

        # Calculate ROUGE-S
        ps, rs, fs, detailss = self._calculate_rouge_s()
        scores.append(
            RougeVerdict(
                metric="rouge_s",
                score=RougeScore(precision=ps, recall=rs, f1_score=fs),
                details=detailss,
                reason="Skip-bigram overlap assessment",
            )
        )

        return scores

    def _calculate_weighted_score(self) -> float:
        """Calculate weighted average of ROUGE scores."""
        total_score = 0.0
        for verdict in self.rouge_scores:
            metric_name = verdict.metric.lower()
            weight = self.weights.get(metric_name, 1 / len(self.rouge_scores))
            total_score += verdict.score.f1_score * weight
        return total_score

    def _generate_final_verdict(self) -> str:
        """Generate final verdict based on ROUGE scores."""
        scores_dict = [score.dict() for score in self.rouge_scores]
        prompt = RougeTemplate.generate_final_verdict(
            scores=scores_dict, final_score=self.score
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]

    def _clean_json_response(self, response: str) -> str:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(prompt=prompt)
        self.total_input_tokens += input_token_count

        if not response:
            raise ValueError("Received an empty response from the model.")

        clean_response = self._clean_json_response(response=response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response

    def _get_detailed_scores(self) -> Dict:
        """Return detailed ROUGE scores analysis."""
        return {
            "overall_score": round(self.score, 3),
            "verdict": self.verdict if self.include_reason else None,
            "detailed_scores": {
                score.metric: {
                    "precision": score.score.precision,
                    "recall": score.score.recall,
                    "f1_score": score.score.f1_score,
                    "details": score.details,
                }
                for score in self.rouge_scores
            },
        }
