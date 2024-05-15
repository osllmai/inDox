import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import BERTScorer
from collections import defaultdict
from sentence_transformers import CrossEncoder
from transformers import pipeline
import textstat

cfg = {"bert_toxic_tokenizer": "unitary/toxic-bert",
       "bert_toxic_model": "unitary/toxic-bert",
       "semantic_similarity": "bert-base-nli-mean-tokens",
       "bert_score_model": "bert-base-uncased",
       "reliability": 'vectara/hallucination_evaluation_model',
       "fairness": "wu981526092/Sentence-Level-Stereotype-Detector"}


ALL_DIMANSIONS = {"BertScore", "Toxicity", "Similarity", "Reliability", "Fairness"}
class Evaluation:
    def __init__(self, dimensions=None, config=cfg):
        if dimensions is None:
            dimensions = ALL_DIMANSIONS

        self.config = config
        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        if not set(dimensions) <= ALL_DIMANSIONS:
            raise RuntimeError('''Please choose correct metrics from ["BertScore","Toxicity","Similarity"]''')
        self.metrics = [eval(dim)(self.config) for dim in dimensions]
        self.result = pd.DataFrame()

    def __call__(self, inputs=None):
        scores = {}
        [scores.update(score(inputs)) for score in self.metrics]
        return scores

    def update(self, inputs):
        result = self.__call__(inputs)
        self.result.append(result, ignore_index=True)
        return self.result

    def reset(self):
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


class Similarity:
    def __init__(self, cfg, inputs=None):
        self.semantic_model = SentenceTransformer(cfg["semantic_similarity"])
        self.vectorizer = TfidfVectorizer()
        self.mlb = MultiLabelBinarizer()

    def __call__(self, inputs):
        self.reference = inputs["context"]
        self.candidate = inputs["answer"]

        bleu = self.bleu_score()
        jaccard = self.jaccard_similarity()
        cosine = self.cosine_similarity()
        semantic = self.semantic_similarity()
        # semantic = None
        scores = {"Blue": bleu,
                  "Jaccard Similarity": jaccard,
                  "Cosine Similarity": cosine,
                  "Semantic": semantic}
        return scores

    def bleu_score(self) -> float:
        """
        Calculate the BLEU score between candidate and reference texts.
        Returns:
        float: The BLEU score.
        """

        reference = [word_tokenize(self.reference)]
        candidate = word_tokenize(self.candidate)

        return sentence_bleu(reference, candidate)

    def jaccard_similarity(self) -> float:
        """
        Calculate the Jaccard similarity between candidate and reference texts.
        Returns:
        float: The Jaccard similarity score.
        """
        reference_tokens = set(word_tokenize(self.reference))
        model_output_tokens = set(word_tokenize(self.candidate))

        binary_reference = self.mlb.fit_transform([reference_tokens])
        binary_model_output = self.mlb.transform([model_output_tokens])

        return jaccard_score(
            binary_reference[0], binary_model_output[0], average="binary"
        )

    def cosine_similarity(self) -> float:
        """
        Calculate the cosine similarity between the TF-IDF vectors of the candidate and reference texts.
        Returns:
        float: The cosine similarity score.
        """
        vectors = self.vectorizer.fit_transform([self.candidate, self.reference])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def semantic_similarity(self) -> float:
        """
        Calculate the semantic similarity between candidate and reference texts using BERT embeddings.

        Returns:
        float: The semantic similarity score.
        """

        embeddings = self.semantic_model.encode(
            [self.candidate, self.reference], convert_to_tensor=True
        )
        return util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0].item()


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
        sentence_pairs = list(zip([context], [answer]))
        results = self.model.predict(sentence_pairs)
        score = [{'hallucination_score': round(score, 2)} for score in results]

        return score[0]


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
        score = result['stereotype_gender']
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
        if not isinstance(answer, list):
            answer = [answer]
        results = {
            "Perplexity": self.calculate_perplexity(answer),
            "ARI": self.calculate_ari(answer),
            "Flesch-Kincaid Grade Level": self.calculate_fk_grade_level(
                answer
            ),
        }
        return results
