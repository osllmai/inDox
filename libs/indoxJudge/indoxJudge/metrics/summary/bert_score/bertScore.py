from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .bertScoreTemplate import BertScoreTemplate


class ModelConfig:
    arbitrary_types_allowed = True


class BertScoreResult(BaseModel):
    precision: float
    recall: float
    f1: float
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config(ModelConfig):
        pass


class BertScoreVerdict(BaseModel):
    score: BertScoreResult
    reason: Optional[str] = None

    class Config(ModelConfig):
        pass


class BertScore:
    def __init__(
        self,
        generated: str,
        reference: str,
        embedding_model_name: str = "bert-base-uncased",
        include_reason: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model=None,
    ):
        self.generated = generated
        self.reference = reference
        self.embedding_model_name = embedding_model_name
        self.include_reason = include_reason
        self.device = device
        self.embedding_model = None
        self.tokenizer = None
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def _initialize_embedding_model(self):
        """Initialize BERT model and tokenizer for embeddings."""
        if self.embedding_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name
            ).to(self.device)
            self.embedding_model.eval()

    def _get_bert_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for input text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)  # Changed from self.model
            embeddings = outputs.last_hidden_state.cpu().numpy()

        return embeddings[0]

    def _call_language_model(self, prompt: str) -> str:
        """Call language model for generating verdict."""
        if self.model is None:
            raise ValueError(
                "Language model not set. Please initialize BertScore with a valid language model"
            )

        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(
            prompt=prompt
        )  # Changed from self.llm
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

    def _calculate_bert_score(self) -> Tuple[float, float, float, Dict]:
        """Calculate BERTScore metrics."""
        self._initialize_embedding_model()

        # Get embeddings
        gen_embeddings = self._get_bert_embeddings(self.generated)
        ref_embeddings = self._get_bert_embeddings(self.reference)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(gen_embeddings, ref_embeddings)

        # Calculate precision (generated → reference)
        precision = np.mean(np.max(similarity_matrix, axis=1))

        # Calculate recall (reference → generated)
        recall = np.mean(np.max(similarity_matrix, axis=0))

        # Calculate F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        details = {
            "avg_precision_similarity": float(precision),
            "avg_recall_similarity": float(recall),
            "token_matches": int(
                np.sum(similarity_matrix > 0.5)
            ),  # Count significant matches
            "max_similarity": float(np.max(similarity_matrix)),
            "min_similarity": float(np.min(similarity_matrix)),
        }

        return precision, recall, f1, details

    def measure(self) -> Dict:
        """Calculate BERTScore and return detailed results."""
        precision, recall, f1, details = self._calculate_bert_score()

        bert_score = BertScoreResult(
            precision=precision, recall=recall, f1=f1, details=details
        )

        result = {
            "overall_score": round(f1, 3),
            "verdict": None,
            "detailed_scores": bert_score.dict(),
        }

        if self.include_reason:
            try:
                result["verdict"] = self._generate_final_verdict(bert_score.dict(), f1)
            except Exception as e:
                logger.error(f"Failed to generate verdict: {str(e)}")
                result["verdict"] = None

        return result

    def _clean_json_response(self, response: str) -> str:
        """
        Cleans the JSON response from the language model by removing markdown code blocks if present.

        :param response: Raw response from the language model
        :return: Cleaned JSON string
        """
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _generate_final_verdict(self, score: Dict, final_score: float) -> str:
        """Generate final verdict based on BERTScore."""
        prompt = BertScoreTemplate.generate_final_verdict(
            score=score, final_score=final_score
        )
        response = self._call_language_model(prompt)

        data = json.loads(response)
        return data["verdict"]
