import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from typing import List, Optional
from .base_classifier import ImageClassifier

class RemoteCLIP(ImageClassifier):
    def __init__(self, model_name: str = "ViT-L-14", pretrained_path: str = "RemoteCLIP-ViT-L-14.pt"):
        """
        Initialize the RemoteCLIP model and its tokenizer.
        :param model_name: Name of the RemoteCLIP model to use.
        :param pretrained_path: Path to the pretrained weights file.
        """
        super().__init__(model_name)
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.load_pretrained_weights(pretrained_path)
        self.model.eval().cuda()

    def load_pretrained_weights(self, pretrained_path: str):
        """
        Load pretrained weights into the model.
        :param pretrained_path: Path to the pretrained weights file.
        """
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)

    def preprocess(self, image: Image, labels: List[str]) -> dict:
        """
        Preprocess the input image and text labels for RemoteCLIP.
        :param image: The input image.
        :param labels: List of text descriptions.
        :return: Dictionary containing preprocessed image and tokenized text.
        """
        image_tensor = self.preprocessor(image.convert('RGB')).unsqueeze(0).cuda()
        text_tensor = self.tokenizer(labels).cuda()
        return {"image": image_tensor, "text": text_tensor}

    def predict(self, inputs: dict) -> np.ndarray:
        """
        Perform prediction using the RemoteCLIP model.
        :param inputs: Preprocessed inputs containing image and text tensors.
        :return: Softmax probabilities as a numpy array.
        """
        image_tensor = inputs["image"]
        text_tensor = inputs["text"]

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs.cpu().numpy()

    def visualize(self, image: Image, labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Visualize the top predicted labels and their probabilities.
        :param image: The input image.
        :param labels: List of text descriptions.
        :param probs: Predicted probabilities for each label.
        :param top: Number of top predictions to display.
        """
        probs = probs[0]  # Extract the first prediction
        top_indices = np.argsort(-probs)[:top]
        top_probs = probs[top_indices]
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
        plt.show()

        # Print the top labels and probabilities
        print([{top_labels[i]: round(top_probs[i], 2)} for i in range(len(top_probs))])

    def classify(self, image: Image, labels: Optional[List[str]] = None, top: int = 5) -> None:
        """
        Full pipeline for classification: preprocess, predict, and visualize.
        :param image: The input image.
        :param labels: Optional list of text descriptions. Uses default labels if not provided.
        :param top: Number of top predictions to display.
        """
        labels = labels or ["a satellite image of an airport", "a satellite image of a university campus", "a satellite image of a lake", "a satellite image of a stadium", "a satellite image of a residential area"]
        inputs = self.preprocess(image, labels)
        probs = self.predict(inputs)
        self.visualize(image, labels, probs, top=top)

