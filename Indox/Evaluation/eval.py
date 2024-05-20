from typing import Dict, Any

import pandas as pd
import textstat
import torch
from bert_score import BERTScorer
from pandas import DataFrame
from sentence_transformers import CrossEncoder
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from .similarity import Similarity
import numpy as np
cfg = {"bert_toxic_tokenizer": "unitary/toxic-bert",
       "bert_toxic_model": "unitary/toxic-bert",
       "semantic_similarity": "sentence-transformers/bert-base-nli-mean-tokens",
       "bert_score_model": "bert-base-uncased",
       "reliability": 'vectara/hallucination_evaluation_model',
       "fairness": "wu981526092/Sentence-Level-Stereotype-Detector",

       }

ALL_DIMANSIONS = ["BertScore", "Toxicity", "Similarity", "Reliability", "Fairness" , "Readibility"]


def to_float(x):
    if isinstance(x, np.ndarray):
        return x.item()  # Convert numpy array to a single float
    return float(x)  # Convert other types to float


class Evaluation:
    def __init__(self, dimensions=None, config=cfg):
        if dimensions is None:
            dimensions = ALL_DIMANSIONS

        self.config = config
        if not isinstance(dimensions, list):
            dimensions = [dimensions]

        self.metrics = [eval(dim)(self.config) for dim in dimensions]
        self.result = pd.DataFrame()

    def __call__(self, inputs=None) -> DataFrame:
        scores = {}
        [scores.update(score(inputs)) for score in self.metrics]
        scores = pd.DataFrame(scores)
        scores = scores.applymap(to_float).T
        return scores

    def update(self, inputs: Any) -> pd.DataFrame:
        result = self.__call__(inputs)
        self.result = pd.concat([self.result, result], ignore_index=True)
        return self.result

    def reset(self) -> None:
        self.result = pd.DataFrame()


class BertScore(BERTScorer):
    def __init__(self, cfg):
        super(BertScore, self).__init__(model_type=cfg["bert_score_model"])

    def __call__(self, inputs):
        answer, context = inputs['answer'], inputs['context']
        if not isinstance(context, list) and len(context) > 1:
            context = [context]
        if not isinstance(answer, list):
            answer = [answer]

        P, R, F1 = self.score(answer, [context], verbose=False)
        scores = {"Precision": P.numpy(),
                  "Recall": R.numpy(),
                  "F1-score": F1.numpy()}
        return scores


class Toxicity:
    def __init__(self, cfg):
        self.bert_tokenizer = BertTokenizer.from_pretrained(cfg["bert_toxic_tokenizer"])
        self.bert_model = BertForSequenceClassification.from_pretrained(cfg["bert_toxic_model"])

    def __call__(self, text):
        # Tokenizing input text and calculating toxicity
        if not isinstance(text, str):
            text = text["answer"]
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        toxicity = probs[
            0, 1
        ].item()  # Assuming that the second class is the 'toxic' class
        scores = {"Toxicity": toxicity}
        return scores


class Reliability:
    """
    This class is designed to evaluate hallucination scores for multiple
    model outputs against reference sentences using a CrossEncoder model.
    """

    def __init__(self, cfg):
        """
        Initializes the SentenceSimilarityEvaluator with a specified model.

        Parameters:
        - model_name (str): The name of the model to be used for hallucination evaluation.
        """
        self.model = CrossEncoder(cfg['reliability'])

    def __call__(self, inputs):
        """
        Predicts similarity scores for each model output in the DataFrame against the reference sentences.

        Parameters:
        - dictionaries: A dictionary containing 'context' and 'answer'.

        Returns:
        - results score in float between 0 and 1  .
        """
        answer, context = inputs['answer'], inputs['context']
        if not isinstance(context, list):
            context = [context]
        relaiabilties = []
        for c in context:
            sentence_pairs = list(zip([c], [answer]))
            results = self.model.predict(sentence_pairs)
            results = [round(score, 2) for score in results]
            relaiabilties.append(results[0])
        score = sum(relaiabilties) / len(relaiabilties)
        score = {'hallucination_score': round(score, 2)}

        return score


class Fairness:
    """
    Sentence-Level Stereotype Classifier

    The Sentence-Level Stereotype Classifier is a transformer-based model developed to detect and classify different types of stereotypes present in the text at the sentence level. It is designed to recognize stereotypical and anti-stereotypical stereotypes towards gender, race, profession, and religion. The model can help in developing applications aimed at mitigating Stereotypical language use and promoting fairness and inclusivity in natural language processing tasks.
    Model Architecture

    The model is built using the pre-trained Distilbert model. It is fine-tuned on MGS Dataset for the task of sentence-level stereotype classification.
    Classes

    The model identifies nine classes, including:

    unrelated: The token does not indicate any stereotype.
    stereotype_gender: The token indicates a gender stereotype.
    anti-stereotype_gender: The token indicates an anti-gender stereotype.
    stereotype_race: The token indicates a racial stereotype.
    anti-stereotype_race: The token indicates an anti-racial stereotype.
    stereotype_profession: The token indicates a professional stereotype.
    anti-stereotype_profession: The token indicates an anti-professional stereotype.
    stereotype_religion: The token indicates a religious stereotype.
    anti-stereotype_religion: The token indicates an anti-religious stereotype.

    """

    def __init__(self, cfg):
        self.model = pipeline("text-classification",
                              model=cfg['fairness'],
                              tokenizer=cfg['fairness'])

    def __call__(self, inputs):
        query = inputs['answer']
        if not isinstance(query, list):
            query = [query]
        result = self.model(query)
        # print(result)
        score = result[0]['score']
        return {"Fairness": score}

class Readibility:
    def __init__(self, cfg=None):
        # Initializing tokenizers and models to be used
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

    def calculate_perplexity(self, text):
        # Tokenizing input text and calculating perplexity
        inputs = self.gpt2_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.item()

    def calculate_ari(self, text):
        # Calculating Automated Readability Index
        return textstat.automated_readability_index(text)

    def calculate_fk_grade_level(self, text):
        # Calculating Flesch-Kincaid Grade Level
        return textstat.flesch_kincaid_grade(text)

    def __call__(self, inputs):
        answer = inputs["answer"]
        if isinstance(answer, list):
            answer = answer[0]
        results = {
            "Perplexity": self.calculate_perplexity(answer),
            "ARI": self.calculate_ari(answer),
            "Flesch-Kincaid Grade Level": self.calculate_fk_grade_level(
                answer
            ),
        }
        return results
