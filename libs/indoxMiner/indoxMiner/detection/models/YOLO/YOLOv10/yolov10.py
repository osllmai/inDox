import os
import cv2
import torch
import numpy as np
import subprocess
import sys
import requests
import glob
import matplotlib.pyplot as plt
import warnings


class YOLOv10:
    def __init__(self, weight_name="yolov10n", device="cuda"):
        """
        Initialize YOLOv10 model.
        Args:
            weight_name (str): Name of YOLOv10 model weights (e.g., 'yolov10n').
            device (str): Device to use ("cuda" or "cpu").
        """
        self.weight_name = weight_name
        self.model_path = self._get_model_weights(weight_name)
        self.device = device if torch.cuda.is_available() else "cpu"

        # Ensure YOLO CLI is installed

    #     self._install_yolo_cli()

    # def _install_yolo_cli(self):
    #     """
    #     Ensures that YOLO CLI is installed.
    #     """
    #     try:
    #         subprocess.run(["yolo", "--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     except FileNotFoundError:
    #         print("YOLO CLI not found. Installing...")
    #         try:
    #             subprocess.run(
    #                 [sys.executable, "-m", "pip", "install", "ultralytics"],
    #                 check=True,
    #             )
    #         except subprocess.CalledProcessError as e:
    #             raise RuntimeError(f"Failed to install YOLO CLI: {e}")

    def _get_model_weights(self, weight_name):
        """
        Downloads the YOLOv10 model weights if not already present.
        Args:
            weight_name (str): Name of the YOLOv10 model weights (e.g., 'yolov10n').
        Returns:
            str: Path to the YOLOv10 model weights file.
        """
        model_path = f"{weight_name}.pt"
        if os.path.exists(model_path):
            return model_path

        print(f"Downloading YOLOv10 model weights: {weight_name}...")
        url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{weight_name}.pt"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model weights downloaded: {model_path}")
            return model_path
        else:
            raise RuntimeError(
                f"Failed to download model weights from {url} (status code: {response.status_code})"
            )

    def detect_objects(self, image_path, conf_threshold=0.25, save_dir="./runs/detect"):
        """
        Perform object detection using YOLOv10 CLI and return the annotated image and detections.
        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for predictions.
            save_dir (str): Directory where results are saved.
        Returns:
            tuple: Annotated image (NumPy array), detection outputs (list of dictionaries).
        """
        # YOLO CLI command
        command = (
            f"yolo task=detect mode=predict model={self.model_path} conf={conf_threshold} "
            f"source={image_path} save=True device={0 if self.device == 'cuda' else 'cpu'}"
        )

        # Run the command with shell=True
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"YOLO command failed with error: {e}")

        # Locate the latest output image in the YOLO save directory
        result_dir = os.path.join(save_dir, "predict")
        if not os.path.exists(result_dir):
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        # Locate the latest result image
        latest_image_path = max(
            glob.glob(f"{result_dir}/*.jpg"), key=os.path.getctime, default=None
        )
        if not latest_image_path:
            raise FileNotFoundError(f"No output image found in {result_dir}")

        # Read the annotated image as a NumPy array
        annotated_image = cv2.imread(latest_image_path)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image

    def visualize_results(self, annotated_image):
        """
        Visualize the YOLOv10 predictions.
        Args:
            annotated_image (np.ndarray): Annotated image with detections.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title("YOLOv10 Object Detection")
        plt.show()
