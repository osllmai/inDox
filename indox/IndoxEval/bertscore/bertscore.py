import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Union, List, Dict

# TODO : Applying different models
class BertScore:
    def __init__(self, model_name: str = "roberta-base", max_length: int = 1024):
        """
        Evaluate the similarity between an actual answer and one or more expected answers
        using a specified pre-trained transformer model.

        This class computes embeddings for the given texts using a pre-trained model
        and calculates similarity scores between the embeddings of the actual answer
        and the expected answers. The similarity is used to derive precision, recall,
        and F1-score metrics, which provide insights into the quality of the actual
        answer compared to the expected answers.

        Parameters:
            model_name (str):
                The identifier for the pre-trained model to be used for generating
                text embeddings. This should be a model name supported by the Hugging
                Face `transformers` library (e.g., 'bert-base-uncased', 'roberta-base').
                The default value is 'bert-base-uncased', which is a standard BERT model
                trained on lowercase English text. Different model names will load
                different model architectures and pre-trained weights, which may impact
                the resulting embeddings and similarity calculations.

            max_length (int):
                The maximum length of input sequences to be processed by the model.
                This determines how many tokens the tokenizer will generate for each
                input text and how long the resulting embeddings will be. If the input
                text exceeds this length, it will be truncated. The default value is
                1024, which accommodates long sequences but may be adjusted based on
                the specific needs of the application or the constraints of the pre-trained
                model.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def get_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def __call__(
        self, llm_answer: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Dict[str, float]:

        if not isinstance(llm_answer, str):
            llm_answer = " ".join(llm_answer)
        llm_answer_embeddings = self.get_embeddings(llm_answer)

        if isinstance(context, str):
            context = [context]

        precisions = []
        recalls = []
        f1_scores = []
        for ctx in context:
            context_embeddings = self.get_embeddings(ctx)

            similarities = np.zeros(
                (len(llm_answer_embeddings), len(context_embeddings))
            )
            for i, a_emb in enumerate(llm_answer_embeddings):
                for j, c_emb in enumerate(context_embeddings):
                    similarities[i, j] = self.cosine_similarity(
                        a_emb.numpy(), c_emb.numpy()
                    )

        precision = np.mean(np.max(similarities, axis=1))
        recall = np.mean(np.max(similarities, axis=0))
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        average_precision = np.mean(precisions) if precisions else 0
        average_recall = np.mean(recalls) if recalls else 0
        average_f1_score = np.mean(f1_scores) if f1_scores else 0

        scores = {
            "Precision": average_precision,
            "Recall": average_recall,
            "F1-score": average_f1_score,
        }

        return scores
