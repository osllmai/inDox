from abc import ABC, abstractmethod
from typing import List, Any
from PIL import Image
import numpy as np

class ImageClassifier(ABC):
    def __init__(self, model_name: str):
        """
        Base class for image classification models.

        :param model_name: Name of the model to use for classification.
        """
        self.model_name = model_name

    @abstractmethod
    def preprocess(self, image: Image, labels: List[str]) -> Any:
        """
        Preprocess the input image and labels.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> np.ndarray:
        """
        Perform prediction on the preprocessed inputs.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def visualize(self, image: Image, labels: List[str], probs: np.ndarray, top: int = 5):
        """
        Visualize the results of the classification.
        """
        pass

    def classify(self, image: Image, labels: List[str]) -> None:
        """
        Full pipeline for classification: preprocess, predict, and visualize.
        """
        inputs = self.preprocess(image, labels)
        probs = self.predict(inputs)
        self.visualize(image, labels, probs)
