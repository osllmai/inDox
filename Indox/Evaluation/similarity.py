import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


class Similarity:
    def __init__(self, cfg, inputs=None):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["semantic_similarity"])
        self.model = AutoModel.from_pretrained(cfg["semantic_similarity"])
        self.vectorizer = TfidfVectorizer()
        self.mlb = MultiLabelBinarizer()

    def __call__(self, inputs):
        self.references = inputs["context"]  # Now it's a list of contexts
        self.candidate = inputs["answer"]
        if not isinstance(self.references, list):
            self.references = [self.references]

        bleu = self.bleu_score()
        jaccard = self.jaccard_similarity()
        cosine = self.cosine_similarity()
        semantic = self.semantic_similarity()
        scores = {
            "BLEU": bleu,
            "Jaccard Similarity": jaccard,
            "Cosine Similarity": cosine,
            "Semantic": semantic
        }
        return scores

    def bleu_score(self) -> float:
        """
        Calculate the BLEU score between candidate and reference texts.
        Returns:
        float: The BLEU score.
        """
        reference = [word_tokenize(ref) for ref in self.references]
        candidate = word_tokenize(self.candidate)
        return sentence_bleu(reference, candidate)

    def jaccard_similarity(self) -> float:
        """
        Calculate the Jaccard similarity between candidate and reference texts.
        Returns:
        float: The Jaccard similarity score.
        """
        reference_tokens = set(word_tokenize(" ".join(self.references)))
        candidate_tokens = set(word_tokenize(self.candidate))
        binary_reference = self.mlb.fit_transform([reference_tokens])
        binary_candidate = self.mlb.transform([candidate_tokens])
        return jaccard_score(binary_reference[0], binary_candidate[0], average="binary")

    def cosine_similarity(self) -> float:
        """
        Calculate the cosine similarity between the TF-IDF vectors of the candidate and reference texts.
        Returns:
        float: The cosine similarity score.
        """
        combined_references = " ".join(self.references)
        vectors = self.vectorizer.fit_transform([self.candidate, combined_references])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def semantic_similarity(self) -> float:
        """
        Calculate the semantic similarity between candidate and reference texts using a pre-trained transformer model.
        Returns:
        float: The semantic similarity score.
        """
        # Tokenize and get embeddings for candidate text
        candidate_inputs = self.tokenizer(self.candidate, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            candidate_outputs = self.model(**candidate_inputs)
            candidate_embedding = candidate_outputs.last_hidden_state.mean(dim=1)

        # Initialize a list to store similarities
        similarities = []

        # Tokenize and get embeddings for each context text, then compute cosine similarity
        with torch.no_grad():
            for ref in self.references:
                ref_inputs = self.tokenizer(ref, return_tensors='pt', truncation=True, padding=True)
                ref_outputs = self.model(**ref_inputs)
                ref_embedding = ref_outputs.last_hidden_state.mean(dim=1)

                similarity = torch.nn.functional.cosine_similarity(candidate_embedding, ref_embedding).item()
                similarities.append(similarity)

        # Average the similarities
        return sum(similarities) / len(similarities)


# Configuration dictionary
cfg = {
    "semantic_similarity": "sentence-transformers/bert-base-nli-mean-tokens"
}


if __name__ == "__main__":
# Example usage
    inputs = {
        "context": [
            "Paris is the capital and most populous city of France.",
            "France's capital, Paris, is a major European city and a global center for art, fashion, and culture.",
            "The capital city of France is Paris, known for its cafe culture and landmarks like the Eiffel Tower."
        ],
        "answer": "The capital of France is Paris."
    }

    similarity = Similarity(cfg)
    scores = similarity(inputs)
    print(scores)
