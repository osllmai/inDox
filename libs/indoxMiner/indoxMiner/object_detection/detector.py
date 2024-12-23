import subprocess
import sys
import torch
from typing import Literal, Optional, Union


class ObjectDetection:
    """
    A factory class for creating object detection model instances.
    """

    def __init__(self):
        """Initialize the ObjectDetection class."""
        self.available_models = ["detectron2", "detr"]

    def _install_detectron2(self) -> bool:
        """
        Install Detectron2 if not already installed.
        Returns:
            bool: True if installation successful or already installed, False otherwise
        """
        try:
            import detectron2

            return True
        except ImportError:
            print("Detectron2 not found. Installing...")
            try:
                # Install torch and torchvision first if not installed
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
                )

                # Install detectron2
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "git+https://github.com/facebookresearch/detectron2.git",
                    ]
                )
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error installing Detectron2: {e}")
                return False

    def load_model(
        self,
        model: Literal["detectron2", "detr"],
        device: Optional[str] = None,
        **kwargs,
    ) -> Union["DETRModel", "Detectron2Model"]:
        """
        Load and return the specified object detection model.

        Args:
            model (str): Model to load ("detectron2" or "detr")
            device (str, optional): Device to use ('cuda' or 'cpu')
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            The loaded model instance
        """
        if model not in self.available_models:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {self.available_models}"
            )

        # Set default device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model == "detectron2":
            if not self._install_detectron2():
                raise RuntimeError("Failed to install Detectron2")

            from .models.detectron2_model import Detectron2Model

            return Detectron2Model(device=device, **kwargs)

        elif model == "detr":
            from .models.detr import DETRModel

            return DETRModel(device=device, **kwargs)
