import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from typing import List, Any
from .base_classifier import ImageClassifier


class CLIPClassifier(ImageClassifier):
    def __init__(self, model_name: str = "openai/CLIP-vit-base-patch32"):
        """
        Initializes the CLIP model for zero-shot image classification.

        Args:
            model_name (str): Hugging Face model name.
        """
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Default labels for general classification
        self.default_labels = ["a cat", "a dog", "a person", "a car", "a tree"]

    def preprocess(self, image: Image.Image, labels: List[str]) -> dict:
        """
        Preprocesses a single image and labels for CLIP classification.

        Args:
            image (Image.Image): Input image.
            labels (List[str]): List of classification labels.

        Returns:
            dict: Preprocessed image and text tensors.
        """
        inputs = self.processor(images=image, text=labels, return_tensors="pt", padding=True)
        return {key: tensor.to(self.device) for key, tensor in inputs.items()}

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Runs inference using the CLIP model and computes classification probabilities.

        Args:
            inputs (dict): Preprocessed input tensors.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().squeeze()
        return probs

    def visualize(self, image: Image.Image, labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Visualizes the classification results.

        Args:
            image (Image.Image): Input image.
            labels (List[str]): List of classification labels.
            probs (np.ndarray): Classification probabilities.
            top (int): Number of top predictions to display.
        """
        top_indices = np.argsort(-probs)[:top]
        top_probs = probs[top_indices]
        top_labels = [labels[i] for i in top_indices]

        # Plot the image and probabilities
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        y = np.arange(len(top_probs))
        plt.grid()
        plt.barh(y, top_probs)
        plt.gca().invert_yaxis()
        plt.yticks(y, top_labels)
        plt.xlabel("Probability")
        plt.title("Classification Results")
        plt.show()

        # Print classification results
        print("Predictions:")
        for i in range(len(top_probs)):
            print(f"- {top_labels[i]}: {round(top_probs[i] * 100, 2)}%")

