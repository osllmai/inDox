import torch
from PIL import Image
from transformers import AutoProcessor, AltCLIPModel
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union
from .base_classifier import ImageClassifier


class AltCLIP(ImageClassifier):
    def __init__(self, model_name: str = "BAAI/AltCLIP"):
        """
        Initialize the AltCLIP model and its processor.
        :param model_name: Name of the AltCLIP model to use.
        """
        super().__init__(model_name)
        self.model = AltCLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.default_labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car"]

    def preprocess(self, images: List[Image.Image], labels: List[str]) -> dict:
        """
        Preprocess a batch of images and text labels for AltCLIP.
        :param images: List of input images.
        :param labels: List of text descriptions.
        :return: Preprocessed inputs.
        """
        inputs = self.processor(text=labels, images=images, return_tensors="pt", padding=True)
        return inputs

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the AltCLIP model for a batch of images.
        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        return logits_per_image.softmax(dim=-1).detach().numpy()

    def visualize(self, images: List[Image.Image], labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Visualize the top predicted labels and their probabilities for a batch of images.
        :param images: List of input images.
        :param labels: List of text descriptions.
        :param probs: Predicted probabilities for each image.
        :param top: Number of top predictions to display per image.
        """
        for i, (image, prob) in enumerate(zip(images, probs)):
            top_indices = np.argsort(-prob)[:top]
            top_probs = prob[top_indices]
            top_labels = [labels[index] for index in top_indices]

            # Plot the image and probabilities
            plt.figure(figsize=(10, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            y = np.arange(len(top_probs))
            plt.grid()
            plt.barh(y, top_probs)
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.yticks(y, top_labels)
            plt.xlabel("Probability")
            plt.title(f"Image {i + 1}")
            plt.show()

            # Print the top labels and probabilities
            print(f"Image {i + 1} predictions:")
            print([{top_labels[j]: round(top_probs[j], 2)} for j in range(len(top_probs))])

    def classify(self, images: Union[Image.Image, List[Image.Image]], labels: Optional[List[str]] = None, top: int = 5) -> None:
        """
        Full pipeline for classification: preprocess, predict, and visualize.

        :param images: Single image or a list of images.
        :param labels: Optional list of text descriptions. Uses default labels if not provided.
        :param top: Number of top predictions to display per image.
        """
        labels = labels or self.default_labels  # Use default labels if none are provided
        if not isinstance(images, list):  # Handle single image input
            images = [images]

        inputs = self.preprocess(images, labels)
        probs = self.predict(inputs)
        self.visualize(images, labels, probs, top=top)
