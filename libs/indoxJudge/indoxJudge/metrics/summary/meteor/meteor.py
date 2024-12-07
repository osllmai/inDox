from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from loguru import logger
import json
import re
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from .meteorTemplate import MeteorTemplate


class ModelConfig:
    arbitrary_types_allowed = True


class MeteorScore(BaseModel):
    score: float
    exact_match: float
    stem_match: float
    synonym_match: float
    fragmentation: float
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config(ModelConfig):
        pass


class MeteorVerdict(BaseModel):
    score: MeteorScore
    reason: Optional[str] = None

    class Config(ModelConfig):
        pass


class Meteor:
    def __init__(
        self,
        summary: str,
        source: str,
        alpha: float = 0.85,
        beta: float = 2.0,
        gamma: float = 0.4,
        weights: List[float] = None,
        include_reason: bool = True,
    ):
        self.generated = summary
        self.reference = source
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights or [
            1.0,
            0.8,
            0.6,
        ]

        self.include_reason = include_reason
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.lemmatizer = WordNetLemmatizer()

    def set_model(self, model):
        self.model = model

    def measure(self) -> Dict:
        """Calculate METEOR score and return detailed results."""
        nltk.download("punkt_tab")
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("averaged_perceptron_tagger_eng")

        score, matches, fragmentation, details = self._calculate_meteor_score()

        meteor_score = MeteorScore(
            score=score,
            exact_match=matches["exact"],
            stem_match=matches["stem"],
            synonym_match=matches["synonym"],
            fragmentation=fragmentation,
            details=details,
        )

        if self.include_reason:
            verdict = self._generate_final_verdict(meteor_score.dict(), score)
        else:
            verdict = None

        return {
            "overall_score": round(score, 3),
            "verdict": verdict,
            "detailed_scores": meteor_score.dict(),
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for METEOR calculation."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep hyphens for compound words
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        # Tokenize
        return word_tokenize(text)

    def _get_wordnet_pos(self, word: str, pos_tag: str) -> str:
        """Convert NLTK POS tag to WordNet POS tag."""
        tag_map = {
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
            "J": wordnet.ADJ,
        }
        return tag_map.get(pos_tag[0], wordnet.NOUN)

    def _get_synonyms(self, word: str, pos: str) -> Set[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        return synonyms

    def _align_words(
        self, generated_tokens: List[str], reference_tokens: List[str]
    ) -> Tuple[Dict[str, float], List[Tuple[int, int]]]:
        """Align words between generated and reference texts."""
        gen_pos = nltk.pos_tag(generated_tokens)
        ref_pos = nltk.pos_tag(reference_tokens)

        matches = {"exact": 0, "stem": 0, "synonym": 0}
        alignments = []
        used_ref = set()

        # First pass: exact matches
        for i, (gen_word, _) in enumerate(gen_pos):
            for j, (ref_word, _) in enumerate(ref_pos):
                if j not in used_ref and gen_word == ref_word:
                    alignments.append((i, j))
                    used_ref.add(j)
                    matches["exact"] += 1
                    break

        # Second pass: stem matches
        for i, (gen_word, gen_tag) in enumerate(gen_pos):
            if any(i == x[0] for x in alignments):
                continue
            gen_stem = self.lemmatizer.lemmatize(
                gen_word, self._get_wordnet_pos(gen_word, gen_tag)
            )
            for j, (ref_word, ref_tag) in enumerate(ref_pos):
                if j in used_ref:
                    continue
                ref_stem = self.lemmatizer.lemmatize(
                    ref_word, self._get_wordnet_pos(ref_word, ref_tag)
                )
                if gen_stem == ref_stem:
                    alignments.append((i, j))
                    used_ref.add(j)
                    matches["stem"] += 1
                    break

        # Third pass: synonym matches
        for i, (gen_word, gen_tag) in enumerate(gen_pos):
            if any(i == x[0] for x in alignments):
                continue
            gen_synonyms = self._get_synonyms(
                gen_word, self._get_wordnet_pos(gen_word, gen_tag)
            )
            for j, (ref_word, ref_tag) in enumerate(ref_pos):
                if j in used_ref:
                    continue
                ref_synonyms = self._get_synonyms(
                    ref_word, self._get_wordnet_pos(ref_word, ref_tag)
                )
                if gen_synonyms & ref_synonyms:  # If there are common synonyms
                    alignments.append((i, j))
                    used_ref.add(j)
                    matches["synonym"] += 1
                    break

        return matches, sorted(alignments)

    def _calculate_fragmentation(
        self, alignments: List[Tuple[int, int]], text_length: int
    ) -> float:
        """Calculate fragmentation penalty."""
        if not alignments:
            return 1.0

        chunks = 1
        for i in range(1, len(alignments)):
            if (
                alignments[i][0] != alignments[i - 1][0] + 1
                or alignments[i][1] != alignments[i - 1][1] + 1
            ):
                chunks += 1

        fragmentation = chunks / len(alignments)
        penalty = self.gamma * (fragmentation**self.beta)
        return penalty

    def _calculate_meteor_score(self) -> Tuple[float, Dict[str, float], float, Dict]:
        """Calculate METEOR score with improved scoring for summaries."""
        generated_tokens = self._preprocess_text(self.generated)
        reference_tokens = self._preprocess_text(self.reference)

        matches, alignments = self._align_words(generated_tokens, reference_tokens)

        # Add smoothing factor for better handling of short texts
        smoothing = 0.1

        total_matches = sum(w * m for w, m in zip(self.weights, matches.values()))
        precision = (total_matches + smoothing) / (len(generated_tokens) + smoothing)
        recall = (total_matches + smoothing) / (len(reference_tokens) + smoothing)

        # Harmonic mean with adjusted alpha
        if precision + recall > 0:
            fmean = (precision * recall) / (
                self.alpha * precision + (1 - self.alpha) * recall
            )
        else:
            fmean = 0.0

        # Reduced fragmentation penalty for summaries
        fragmentation = self._calculate_fragmentation(alignments, len(generated_tokens))
        fragmentation = fragmentation * 0.75  # Reduce the penalty impact

        score = fmean * (1 - fragmentation)

        details = {
            "generated_length": len(generated_tokens),
            "reference_length": len(reference_tokens),
            "precision": precision,
            "recall": recall,
            "fmean": fmean,
            "total_matches": total_matches,
            "alignments": len(alignments),
        }

        return score, matches, fragmentation, details

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from language model."""
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        """Call language model for generating verdict."""
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

    def _generate_final_verdict(self, score: Dict, final_score: float) -> str:
        """Generate final verdict based on METEOR scores."""
        prompt = MeteorTemplate.generate_final_verdict(
            score=score, final_score=final_score
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]
