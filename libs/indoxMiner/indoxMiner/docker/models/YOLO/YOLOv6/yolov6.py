import os
import subprocess
import urllib.request
import torch
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt


class YOLOv6:
    def __init__(self, model_name="yolov6s", device="cuda"):
        """
        Initialize the YOLOv6 model and download weights if necessary.
        Args:
            model_name (str): Name of the YOLOv6 model (e.g., 'yolov6n', 'yolov6s', 'yolov6m').
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.weights_path = f"{model_name}.pt"
        self.weights_url = f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{self.weights_path}"

        # Ensure YOLOv6 is installed
        self._install_yolov6()

        # Download weights if necessary
        self._download_weights()

    def _install_yolov6(self):
        """
        Install YOLOv6 from the official repository if not already installed.
        """
        if not os.path.exists("YOLOv6"):
            print("Cloning YOLOv6 repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/meituan/YOLOv6"], check=True
            )
            print("YOLOv6 repository cloned.")
        os.chdir("YOLOv6")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        os.chdir("..")

    def _download_weights(self):
        """
        Download the YOLOv6 weights file if it doesn't exist.
        """
        if not os.path.exists(self.weights_path):
            print(
                f"Downloading weights for {self.model_name} from {self.weights_url}..."
            )
            urllib.request.urlretrieve(self.weights_url, self.weights_path)
            print("Download complete.")
        else:
            print(f"Weights for {self.model_name} already exist.")

    def detect_objects(self, image_path, conf_threshold=0.25):
        """
        Detect objects in an image using YOLOv6 inference.
        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for detections.
        Returns:
            np.ndarray: Annotated image with detections.
        """
        # Ensure YOLOv6 repository is active
        os.chdir("YOLOv6")

        # Output directory for inference results
        output_dir = "runs/inference/"

        # YOLOv6 CLI command
        command = [
            "python",
            "tools/infer.py",
            "--weights",
            f"../{self.weights_path}",
            "--source",
            image_path,
            "--save-dir",
            output_dir,
            "--conf-thres",
            str(conf_threshold),
        ]

        print("Running YOLOv6 inference...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("YOLOv6 Output:", result.stdout)  # Log YOLOv6 output for debugging
        print("YOLOv6 Error:", result.stderr)

        # Check for output images
        output_images = glob.glob(
            os.path.join(output_dir, "*.*")
        )  # Match all file types
        if not output_images:
            raise FileNotFoundError(
                f"No output images found after inference. Check logs for more details."
            )

        # Read the annotated image
        annotated_image_path = output_images[0]
        annotated_image = cv2.imread(annotated_image_path)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Return to the original directory
        os.chdir("..")

        return annotated_image

    def visualize_results(self, annotated_image):
        """
        Visualize and display annotated objects in the image.
        Args:
            annotated_image (np.ndarray): Annotated image with detections.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.title(f"YOLOv6 ({self.model_name}) Detection Results")
        plt.axis("off")
        plt.show()
