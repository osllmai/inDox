import os
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO


class YOLOv7:
    def __init__(self, model_path="/app/yolov7.pt", device="cuda"):
        """
        Initialize the YOLOv7 model.

        Args:
            model_path (str): Path to the YOLOv7 model weights.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path

        # Load YOLOv7 model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def detect_objects(self, image_path, conf_threshold=0.25, save_dir="/app/results"):
        """
        Detect objects in an image using YOLOv7.

        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for detections.
            save_dir (str): Directory to save annotated images and results.

        Returns:
            tuple: Annotated image (NumPy array), detection outputs (list of dictionaries).
        """
        # Perform inference
        results = self.model.predict(source=image_path, conf=conf_threshold, save=True)

        # Get the annotated image path
        annotated_image_path = os.path.join(save_dir, os.path.basename(image_path))
        if not os.path.exists(annotated_image_path):
            raise FileNotFoundError("Annotated image not found in save directory.")

        # Load the annotated image
        annotated_image = cv2.imread(annotated_image_path)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes.xyxy:
                detections.append(
                    {
                        "bbox": box[:4].tolist(),
                        "confidence": box[4].item(),
                        "class_id": box[5].item(),
                    }
                )

        return annotated_image, detections

    def visualize_results(self, annotated_image, detections):
        """
        Visualize and display annotated objects in the image.

        Args:
            annotated_image (np.ndarray): Annotated image with detections.
            detections (list): List of detection outputs.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.title(f"YOLOv7 Detection Results")
        plt.axis("off")
        plt.show()
