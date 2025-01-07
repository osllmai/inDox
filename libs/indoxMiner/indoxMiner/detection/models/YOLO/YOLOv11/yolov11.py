import os
import subprocess
import glob
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import sys


class YOLOv11:
    def __init__(self, model_name="yolo11n", device="cuda"):
        """
        Initialize YOLOv11 model.
        Args:
            model_name (str): Name of YOLOv11 model weights (e.g., 'yolo11n').
            device (str): Device to use ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = self._get_model_weights(model_name)

        # Ensure YOLO CLI is installed
        self._install_yolo_cli()

    def _install_yolo_cli(self):
        """
        Ensures that YOLO CLI is installed.
        """
        try:
            subprocess.run(["yolo", "--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("YOLO CLI not found. Installing...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "ultralytics"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install YOLO CLI: {e}")

    def _get_model_weights(self, model_name):
        """
        Downloads the YOLOv11 model weights if not already present.
        Args:
            model_name (str): Name of the YOLOv11 model weights (e.g., 'yolo11n').
        Returns:
            str: Path to the YOLOv11 model weights file.
        """
        model_path = f"{model_name}.pt"
        if os.path.exists(model_path):
            return model_path

        print(f"Downloading YOLOv11 model weights: {model_name}...")
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model weights downloaded: {model_path}")
            return model_path
        else:
            raise RuntimeError(f"Failed to download model weights from {url} (status code: {response.status_code})")

    def detect_objects(self, image_path, conf_threshold=0.25, save_dir="./runs/detect"):
        """
        Perform object detection using YOLOv11 CLI.
        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for predictions.
            save_dir (str): Directory where results are saved.
        Returns:
            str: Path to the saved output image.
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

        latest_image = max(glob.glob(f"{result_dir}/*.jpg"), key=os.path.getctime, default=None)
        if not latest_image:
            raise FileNotFoundError(f"No output image found in {result_dir}")

        return latest_image

    def visualize_results(self, result_image_path):
        """
        Visualize the YOLOv11 predictions.
        Args:
            result_image_path (str): Path to the output image saved by YOLOv11.
        """
        if not os.path.exists(result_image_path):
            raise FileNotFoundError(f"Result image not found: {result_image_path}")

        img = mpimg.imread(result_image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("YOLOv11 Object Detection")
        plt.show()
