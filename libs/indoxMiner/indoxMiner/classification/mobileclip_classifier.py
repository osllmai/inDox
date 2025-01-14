import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mobileclip
from typing import List, Optional, Union


class MobileCLIPClassifier(ImageClassifier):
    def __init__(self, model_name: str = "mobileclip_s0", pretrained_path: str = "mobileclip_s0.pt"):
        """
        Initialize the MobileCLIP model and its tokenizer.
        :param model_name: Name of the MobileCLIP model to use.
        :param pretrained_path: Path to the pretrained weights file.
        """
        super().__init__(model_name)
        self.model, _, self.preprocessor = mobileclip.create_model_and_transforms(
            model_name, pretrained=pretrained_path
        )
        self.tokenizer = mobileclip.get_tokenizer(model_name)
        self.default_labels = ["a diagram", "a dog", "a cat", "a bird", "a car"]

    def preprocess(self, images: List[Image.Image], labels: List[str]) -> dict:
        """
        Preprocess a batch of images and text labels for MobileCLIP.
        :param images: List of input images.
        :param labels: List of text descriptions.
        :return: Preprocessed inputs as a dictionary.
        """
        image_tensors = torch.stack([self.preprocessor(image.convert("RGB")) for image in images])
        text_tensors = self.tokenizer(labels)
        return {"image": image_tensors, "text": text_tensors}

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the MobileCLIP model for a batch of images.
        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        image_tensors = inputs["image"]
        text_tensors = inputs["text"]

        with torch.no_grad(), torch.cuda.amp.autocast():
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
