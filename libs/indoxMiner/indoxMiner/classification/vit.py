from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union
from .base_classifier import ImageClassifier


class ViT(ImageClassifier):
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the ViT model and processor.
        :param model_name: Name of the ViT model to use.
        """
        super().__init__(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        # Predefined classes for ImageNet (retrieved from model configuration)
        self.predefined_classes = list(self.model.config.id2label.values())

    def preprocess(self, images: List[Image.Image]) -> dict:
        """
        Preprocess a batch of input images for the ViT model.
        :param images: List of input images.
        :return: Preprocessed inputs as a dictionary.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the ViT model for a batch of images.
        :param inputs: Preprocessed inputs.
        :return: Softmax probabilities as a numpy array.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return logits.softmax(dim=-1).detach().numpy()

    def visualize(self, images: List[Image.Image], probs: np.ndarray, top: int = 5):
        """
        Visualize the top predicted ImageNet classes and probabilities for a batch of images.
        :param images: List of input images.
        :param probs: Predicted probabilities for each image.
        :param top: Number of top predictions to display per image.
        """
        for i, (image, prob) in enumerate(zip(images, probs)):
            top_labels = np.argsort(-prob)[:top]
            top_probs = prob[top_labels]
            top_classes = [self.model.config.id2label[idx] for idx in top_labels]

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
            plt.yticks(y, top_classes)
            plt.xlabel("Probability")
            plt.title(f"Image {i + 1}")
            plt.show()

            # Print the top classes and probabilities
            print(f"Image {i + 1} predictions:")
            print([{top_classes[j]: round(top_probs[j], 2)} for j in range(len(top_probs))])

    def classify(self, images: Union[Image.Image, List[Image.Image]], top: int = 5) -> None:
        """
        Full pipeline for classification: preprocess, predict, and visualize.

        :param images: Single image or a list of images.
        :param top: Number of top predictions to display per image.
        """
        if not isinstance(images, list):  # Handle single image input
            images = [images]

        inputs = self.preprocess(images)
        probs = self.predict(inputs)
        self.visualize(images, probs, top=top)
