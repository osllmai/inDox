import os
import argparse
from typing import Optional, List, Union
import torch
from pathlib import Path
from models import *


class ModelFactory:
    """Factory class to create and manage different object detection models."""

    def __init__(self):
        self.models = {}
        self._register_models()

    def _register_models(self):
        """Register all available models."""
        self.models = {
            "Detectron2": Detectron2,
            "DETR": DETR,
            "DETRCLIP": DETRCLIP,
            "GroundingDINO": GroundingDINO,
            "Kosmos2": Kosmos2,
            "OWLVit": OWLVit,
            "RTDETR": RTDETR,
            "SAM2": SAM2,
            "YOLOv5": YOLOv5,
            "YOLOv6": YOLOv6,
            "YOLOv7": YOLOv7,
            "YOLOv8": YOLOv8,
            "YOLOv10": YOLOv10,
            "YOLOv11": YOLOv11,
            "YOLOX": YOLOX,
        }

    def get_model(self, model_name: str, **kwargs) -> object:
        """
        Get an instance of the specified model.

        Args:
            model_name: Name of the model to instantiate
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            Instantiated model object
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(self.models.keys())}"
            )

        return self.models[model_name](**kwargs)


class InferenceEngine:
    """Main inference engine that handles multiple models and batch processing."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the inference engine.

        Args:
            model_name: Name of the model to use
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_factory = ModelFactory()
        self.model = self.model_factory.get_model(model_name, device=self.device)

    def process_single_image(
        self, image_path: str, score_threshold: float = 0.5, visualize: bool = False
    ):
        """
        Process a single image.

        Args:
            image_path: Path to the image file
            score_threshold: Confidence threshold for detections
            visualize: Whether to visualize the results

        Returns:
            Detection results
        """
        results = self.model.detect_objects(image_path, score_threshold=score_threshold)

        if visualize:
            self.model.visualize_results(results)

        return results

    def process_batch(
        self,
        image_paths: List[str],
        score_threshold: float = 0.5,
        visualize: bool = False,
    ):
        """
        Process a batch of images.

        Args:
            image_paths: List of paths to image files
            score_threshold: Confidence threshold for detections
            visualize: Whether to visualize the results

        Returns:
            List of detection results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.process_single_image(
                    image_path, score_threshold=score_threshold, visualize=visualize
                )
                results.append({"path": image_path, "detections": result})
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({"path": image_path, "error": str(e)})

        return results


def main():
    parser = argparse.ArgumentParser(description="Object Detection Inference")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to run inference on"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize results")

    args = parser.parse_args()

    # Initialize inference engine
    engine = InferenceEngine(args.model, device=args.device)

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        results = engine.process_single_image(
            str(input_path), score_threshold=args.threshold, visualize=args.visualize
        )
        print(f"Results for {input_path}:")
        print(results)
    else:
        # Directory of images
        image_paths = [
            str(p) for p in input_path.glob("*.jpg") + input_path.glob("*.png")
        ]
        results = engine.process_batch(
            image_paths, score_threshold=args.threshold, visualize=args.visualize
        )
        print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()
