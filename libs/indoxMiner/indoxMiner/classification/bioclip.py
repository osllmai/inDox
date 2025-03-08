import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from typing import List, Optional, Union
from .base_classifier import ImageClassifier


class BioCLIP(ImageClassifier):
    def __init__(self, model_name: str = "hf-hub:imageomics/bioclip"):
        """
        Initialize the BioCLIP model and its tokenizer.
        :param model_name: Name of the BioCLIP model to use.
        """
        super().__init__(model_name)
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # Default labels (can be customized based on BioCLIP's domain)
        self.default_labels = ["a cell", "a tissue sample", "a protein structure", "a molecule", "a diagram"]

    def preprocess(self, images: List[Image.Image], labels: List[str]) -> dict:
        """
        Preprocess a batch of images and text labels for BioCLIP.
        :param images: List of input images.
        :param labels: List of text descriptions.
        :return: Preprocessed inputs.
        """
        image_tensors = torch.stack([self.preprocess_val(image.convert("RGB")) for image in images])
        text_tensor = self.tokenizer(labels)
        return {"image": image_tensors, "text": text_tensor}

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the BioCLIP model for a batch of images.
        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        image_tensors = inputs["image"]
        text_tensors = inputs["text"]

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tensors)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs.detach().numpy()

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
        if not isinstance(images, list):  # Handle a single image input
            images = [images]

        inputs = self.preprocess(images, labels)
        probs = self.predict(inputs)
        self.visualize(images, labels, probs, top=top)
