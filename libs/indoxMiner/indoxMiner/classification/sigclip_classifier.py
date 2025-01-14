from transformers import AutoProcessor, AutoModel
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Union
import numpy as np
from PIL import Image


class SigCLIPClassifier(ImageClassifier):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Default labels (can be customized)
        self.default_labels = [
            "cat", "dog", "wolf", "tiger", "man", "horse", "frog", "tree", "house", "computer"
        ]

    def preprocess(self, images: List[Image.Image], labels: List[str]) -> dict:
        """
        Preprocess a batch of images and text labels.

        :param images: List of input images.
        :param labels: List of text labels.
        :return: Dictionary containing preprocessed image and text tensors.
        """
        text_descriptions = [f"This is a photo of a {label}" for label in labels]
        inputs = self.processor(text=text_descriptions, images=images, padding="max_length", return_tensors="pt")
        return inputs

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the SigCLIP model for a batch of images.

        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        with torch.no_grad():
            self.model.config.torchscript = False
            results = self.model(**inputs)
        logits_per_image = results["logits_per_image"]  # Image-text similarity scores
        return logits_per_image.softmax(dim=1).detach().numpy()

    def visualize(self, images: List[Image.Image], labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Visualize the classification results for a batch of images.

        :param images: List of input images.
        :param labels: List of text labels.
        :param probs: Predicted probabilities for each label per image.
        :param top: Number of top predictions to visualize for each image.
        """
        for i, (image, prob) in enumerate(zip(images, probs)):
            top_labels = np.argsort(-prob)[:top]
            top_probs = prob[top_labels]

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
            plt.yticks(y, [labels[index] for index in top_labels])
            plt.xlabel("Probability")
            plt.title(f"Image {i + 1}")
            plt.show()

            # Print the top labels and probabilities
            print(f"Image {i + 1} predictions:")
            print([{labels[index]: round(prob, 2)} for index, prob in zip(top_labels, top_probs)])

    def classify(self, images: Union[Image.Image, List[Image.Image]], labels: Optional[List[str]] = None, top: int = 5) -> None:
        """
        Full pipeline for classification: preprocess, predict, and visualize.

        :param images: Single image or a list of images.
        :param labels: Optional list of labels. Uses default labels if not provided.
        :param top: Number of top predictions to display for each image.
        """
        labels = labels or self.default_labels  # Use default labels if none are provided
        if not isinstance(images, list):  # Handle a single image as input
            images = [images]

        inputs = self.preprocess(images, labels)
        probs = self.predict(inputs)
        self.visualize(images, labels, probs, top=top)
