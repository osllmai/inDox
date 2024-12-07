import difflib
import math
import numpy as np
import re
import string
from typing import List, Union
import warnings
from transformers import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint at")
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
warnings.filterwarnings("ignore", message="This function will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library.")



class Gruen:
    def __init__(self, candidates: Union[str, List[str]]):
        """
        Initialize the TextEvaluator with candidate texts.

        Parameters:
        candidates (Union[str, List[str]]): The candidate text(s) to evaluate.
        """
        import torch
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from indoxJudge.utils import nltk_download
        nltk_download()
        if isinstance(candidates, str):
            candidates = [candidates]
        self.candidates = candidates
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_candidates(self) -> List[List[str]]:
        """
        Preprocess the candidate texts.

        Returns:
        List[List[str]]: A list of lists, each containing sentences from the preprocessed candidate texts.
        """
        from nltk import sent_tokenize

        processed_candidates = []
        for candidate in self.candidates:
            candidate = self._clean_text(candidate)
            sentences = sent_tokenize(candidate)
            processed_sentences = [
                sentence
                for sentence in sentences
                if len(sentence.translate(str.maketrans("", "", string.punctuation)).split()) > 1
            ]
            processed_candidates.append(processed_sentences)
        return processed_candidates

    def _clean_text(self, text: str) -> str:
        """
        Clean the input text by normalizing punctuation and whitespace.

        Parameters:
        text (str): The text to clean.

        Returns:
        str: The cleaned text.
        """

        text = text.strip()
        text = ". ".join(text.split("\n\n"))
        text = ". ".join(text.split("\n"))
        text = ".".join(text.split(".."))
        text = ". ".join(text.split("."))
        text = ". ".join(text.split(". . "))
        text = ". ".join(text.split(".  . "))
        while "  " in text:
            text = " ".join(text.split("  "))
        text = re.sub(r"(\d+)\. (\d+)", "UNK", text)
        return text.strip()

    def get_lm_score(self, sentences: List[List[str]]) -> List[float]:
        """
        Calculate the language model (LM) score for the given sentences.

        Parameters:
        sentences (List[List[str]]): List of lists containing sentences.

        Returns:
        List[float]: The LM scores for the sentences.
        """
        import torch
        from transformers import BertForMaskedLM, BertTokenizer
        from tqdm import tqdm

        model_name = "bert-base-cased"
        model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_name)

        lm_scores = []
        for sentence in tqdm(sentences):
            if not sentence:
                lm_scores.append(0.0)
                continue
            score = sum(
                self._score_sentence(s, tokenizer, model) for s in sentence
            ) / len(sentence)
            lm_scores.append(score)
        return lm_scores

    def _score_sentence(self, sentence: str, tokenizer, model) -> float:
        """
        Calculate the score for a single sentence using the language model.

        Parameters:
        sentence (str): The sentence to score.
        tokenizer: The tokenizer for the language model.
        model: The language model.

        Returns:
        float: The calculated score for the sentence.
        """
        import torch
        import math

        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > 510:
            tokens = tokens[:510]
        input_ids = torch.tensor(tokenizer.encode(tokens)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids)[0]
        return math.exp(loss.item())

    def get_cola_score(self, sentences: List[List[str]]) -> List[float]:
        """
        Calculate the CoLA score for grammatical acceptability.

        Parameters:
        sentences (List[List[str]]): List of lists containing sentences.

        Returns:
        List[float]: The CoLA scores for the sentences.
        """
        model_name = "textattack/bert-base-uncased-CoLA"
        tokenizer, model = self._load_pretrained_cola_model(model_name)
        flat_sentences = [sentence for sublist in sentences for sentence in sublist]
        sentence_lengths = [len(sublist) for sublist in sentences]
        cola_scores = self._evaluate_cola(model, flat_sentences, tokenizer, model_name)
        return self._convert_sentence_to_paragraph_scores(cola_scores, sentence_lengths)

    def _load_pretrained_cola_model(self, model_name: str):

        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        import torch

        config = BertConfig.from_pretrained(
            model_name, num_labels=2, finetuning_task="CoLA"
        )
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(
            model_name, config=config
        ).to(self.device)
        model.eval()
        return tokenizer, model

    def _evaluate_cola(
            self, model, candidates: List[str], tokenizer, model_name: str
    ) -> List[float]:
        import torch
        from tqdm import tqdm

        dataset = self._load_and_cache_examples(candidates, tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=max(1, torch.cuda.device_count()),
        )
        preds = None
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if model_name.split("-")[0] != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if model_name.split("-")[0] in ["bert", "xlnet"]
                        else None
                    )
                outputs = model(**inputs)
                logits = outputs[1].detach().cpu().numpy()
                preds = logits if preds is None else np.append(preds, logits, axis=0)
        return preds[:, 1].tolist()

    def _load_and_cache_examples(self, candidates: List[str], tokenizer):
        import torch
        from transformers import glue_convert_examples_to_features
        from transformers.data.processors.utils import InputExample

        examples = [
            InputExample(guid=str(i), text_a=c) for i, c in enumerate(candidates)
        ]
        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=["0", "1"],
            max_length=128,
            output_mode="classification",
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([0 for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [[0.0] * 128 for _ in features], dtype=torch.long
        )
        return torch.utils.data.TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )

    def _convert_sentence_to_paragraph_scores(
            self, sentence_scores: List[float], sent_length: List[int]
    ) -> List[float]:
        paragraph_scores = []
        pointer = 0
        for length in sent_length:
            if length == 0:
                paragraph_scores.append(0.0)
                continue
            temp_scores = sentence_scores[pointer: pointer + length]
            paragraph_scores.append(sum(temp_scores) / length)
            pointer += length
        return paragraph_scores

    def get_grammaticality_score(
            self, processed_candidates: List[List[str]]
    ) -> List[float]:
        """
        Calculate the grammaticality score using LM and CoLA scores.

        Parameters:
        processed_candidates (List[List[str]]): Preprocessed candidate texts.

        Returns:
        List[float]: The grammaticality scores.
        """
        lm_scores = self.get_lm_score(processed_candidates)
        cola_scores = self.get_cola_score(processed_candidates)
        grammaticality_scores = [
            1.0 * math.exp(-0.5 * lm) + 1.0 * cola
            for lm, cola in zip(lm_scores, cola_scores)
        ]
        return [max(0, score / 8.0 + 0.5) for score in grammaticality_scores]

    def get_redundancy_score(self, all_summary: List[List[str]]) -> List[float]:
        """
        Calculate the redundancy score for the summaries.

        Parameters:
        all_summary (List[List[str]]): Summarized texts to evaluate.

        Returns:
        List[float]: The redundancy scores.
        """
        from difflib import SequenceMatcher

        redundancy_scores = [0.0] * len(all_summary)
        for i, summary in enumerate(all_summary):
            if len(summary) == 1:
                continue
            flag = sum(
                self._if_two_sentences_redundant(summary[j].strip(), summary[k].strip())
                for j in range(len(summary) - 1)
                for k in range(j + 1, len(summary))
            )
            redundancy_scores[i] += -0.1 * flag
        return redundancy_scores

    def _if_two_sentences_redundant(self, sentence_a: str, sentence_b: str) -> int:
        """
        Determine if two sentences are redundant.

        Parameters:
        sentence_a (str): First sentence.
        sentence_b (str): Second sentence.

        Returns:
        int: Redundancy flag (0 or higher).
        """
        if (
                sentence_a == sentence_b
                or sentence_a in sentence_b
                or sentence_b in sentence_a
        ):
            return 4
        redundancy_flag = 0
        sentence_a_split, sentence_b_split = sentence_a.split(), sentence_b.split()
        if max(len(sentence_a_split), len(sentence_b_split)) >= 5:
            longest_common_substring = difflib.SequenceMatcher(
                None, sentence_a, sentence_b
            ).find_longest_match(0, len(sentence_a), 0, len(sentence_b))
            LCS_length = longest_common_substring.size
            if LCS_length > 0.8 * min(len(sentence_a), len(sentence_b)):
                redundancy_flag += 1
            LCS_word_length = len(
                sentence_a[
                longest_common_substring[0]: (
                        longest_common_substring[0] + LCS_length
                )
                ]
                .strip()
                .split()
            )
            if LCS_word_length > 0.8 * min(
                    len(sentence_a_split), len(sentence_b_split)
            ):
                redundancy_flag += 1
            edit_distance = self._levenshtein_distance(sentence_a, sentence_b)
            if edit_distance < 0.6 * max(len(sentence_a), len(sentence_b)):
                redundancy_flag += 1
            common_word_count = len(
                [word for word in sentence_a_split if word in sentence_b_split]
            )
            if common_word_count > 0.8 * min(
                    len(sentence_a_split), len(sentence_b_split)
            ):
                redundancy_flag += 1
        return redundancy_flag

    def _levenshtein_distance(self, string1: str, string2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Parameters:
        string1 (str): First string.
        string2 (str): Second string.

        Returns:
        int: Levenshtein distance between the two strings.
        """
        if len(string1) < len(string2):
            return self._levenshtein_distance(string2, string1)
        if len(string2) == 0:
            return len(string1)

        previous_row = range(len(string2) + 1)
        for i, char1 in enumerate(string1):
            current_row = [i + 1]
            for j, char2 in enumerate(string2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (char1 != char2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_focus_score(self, all_summary: List[List[str]]) -> List[float]:
        """
        Calculate the focus score based on sentence similarity using NLTK.

        Parameters:
        all_summary (List[List[str]]): Summarized texts to evaluate.

        Returns:
        List[float]: The focus scores.
        """
        all_scores = []
        for summary in all_summary:
            if len(summary) == 1:
                all_scores.append([1.0])
                continue
            scores = []
            for j in range(1, len(summary)):
                similarity = self._sentence_similarity(summary[j - 1], summary[j])
                scores.append(1.0 / (1.0 + math.exp(-similarity + 7)))
            all_scores.append(scores)
        return [0.0 if not scores or min(scores) < 0.05 else -0.1 for scores in all_scores]

    def _sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calculate the similarity between two sentences using NLTK.

        Parameters:
        sentence1 (str): First sentence.
        sentence2 (str): Second sentence.

        Returns:
        float: Similarity score between the two sentences.
        """
        # Tokenize and lemmatize the sentences
        from nltk.tokenize import word_tokenize
        words1 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence1) if word.lower() not in self.stop_words]
        words2 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence2) if word.lower() not in self.stop_words]

        # Find the union of words
        all_words = set(words1 + words2)

        # Create word vectors
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))

        if magnitude1 * magnitude2 == 0:
            return 0.0
        else:
            return dot_product / (magnitude1 * magnitude2)

    def measure(self) -> List[float]:
        """
        Calculate the GRUEN score for the candidates.

        Returns:
        List[float]: The GRUEN scores.
        """
        processed_candidates = self.preprocess_candidates()
        grammaticality_scores = self.get_grammaticality_score(processed_candidates)
        redundancy_scores = self.get_redundancy_score(processed_candidates)
        focus_scores = self.get_focus_score(processed_candidates)
        gruen_scores = [
            min(1, max(0, sum(scores)))
            for scores in zip(grammaticality_scores, redundancy_scores, focus_scores)
        ]
        gruen_scores = [round(score, 2) for score in gruen_scores]
        return gruen_scores
