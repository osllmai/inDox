import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
from .base_classifier import ImageClassifier

class BiomedCLIPClassifier(ImageClassifier):
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        """
        Initialize the BiomedCLIP model and its tokenizer.
        :param model_name: Name of the BiomedCLIP model to use.
        """
        super().__init__(model_name)
        # Rename the transform function to avoid clashing with self.preprocess
        self.model, self.image_transform = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.default_labels = [
            "adenocarcinoma histopathology",
            "brain MRI",
            "covid line chart",
            "squamous cell carcinoma histopathology",
            "immunohistochemistry histopathology",
            "bone X-ray",
            "chest X-ray",
            "pie chart",
            "hematoxylin and eosin histopathology"
        ]

    def preprocess(self, images: List[Image.Image], labels: List[str]) -> dict:
        """
        Preprocess a batch of images and text labels for BiomedCLIP.
        :param images: List of input images.
        :param labels: List of text descriptions.
        :return: Preprocessed inputs.
        """
        try:
            # Use self.image_transform (not self.preprocess) for image transformations
            image_tensors = torch.stack([
                self.image_transform(image.convert("RGB"))
                for image in images
            ]).to(self.device)
        except TypeError as e:
            raise RuntimeError(
                "Error in image preprocessing pipeline. Ensure images are "
                "compatible with the preprocessing transforms."
            ) from e

        # Tokenize the text descriptions
        text_tensors = self.tokenizer(
            [f"this is a photo of {label}" for label in labels]
        ).to(self.device)

        return {"image": image_tensors, "text": text_tensors}

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the BiomedCLIP model for a batch of images.
        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        image_tensors = inputs["image"]
        text_tensors = inputs["text"]

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(
                image_tensors, text_tensors
            )
            # Compute similarity and apply softmax
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        return logits.detach().cpu().numpy()

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

            print(f"Image {i + 1} predictions:")
            print([{top_labels[j]: round(top_probs[j], 2)} for j in range(len(top_probs))])

    def classify(self, images: Union[Image.Image, List[Image.Image]], labels: Optional[List[str]] = None, top: int = 5):
        """
        Full pipeline for classification: preprocess, predict, and visualize.

        :param images: Single image or a list of images.
        :param labels: Optional list of text descriptions. Uses default labels if not provided.
        :param top: Number of top predictions to display per image.
        """
        labels = labels or self.default_labels
        if not isinstance(images, list):
            images = [images]

        inputs = self.preprocess(images, labels)
        probs = self.predict(inputs)
        self.visualize(images, labels, probs, top=top)