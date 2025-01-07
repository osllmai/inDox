import os
import subprocess
import urllib.request
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class YOLOv7:
    def __init__(self, model_name="yolov7", device="cuda"):
        """
        Initialize the YOLOv7 model and download weights if necessary.
        Args:
            model_name (str): Name of the YOLOv7 model (e.g., 'yolov7', 'yolov7x').
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.weights_path = f"{model_name}.pt"
        self.weights_url = f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{self.weights_path}"

        # Ensure YOLOv7 is installed
        self._install_yolov7()

        # Download weights if necessary
        self._download_weights()

    def _install_yolov7(self):
        if not os.path.exists("yolov7"):
            print("Cloning YOLOv7 repository...")
            subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7"], check=True)
            print("YOLOv7 repository cloned.")
        os.chdir("yolov7")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        os.chdir("..")

    def _download_weights(self):
        if not os.path.exists(self.weights_path):
            print(f"Downloading weights for {self.model_name} from {self.weights_url}...")
            urllib.request.urlretrieve(self.weights_url, self.weights_path)
            print("Download complete.")
        else:
            print(f"Weights for {self.model_name} already exist.")

    def detect_objects(self, image_path, conf_threshold=0.25):
        """
        Detect objects in an image using YOLOv7 inference.

        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): Confidence threshold for detections.

        Returns:
            dict: A dictionary containing the annotated image and detections.
        """
        # Save the original working directory
        original_cwd = os.getcwd()

        try:
            # Change to YOLOv7 directory
            os.chdir("yolov7")

            # YOLOv7 CLI command
            command = [
                "python", "detect.py",
                "--weights", f"../{self.weights_path}",
                "--source", image_path,
                "--conf-thres", str(conf_threshold),
                "--save-txt", "--save-conf"
            ]

            print("Running YOLOv7 inference...")
            subprocess.run(command, check=True, capture_output=True, text=True)

            # Determine the latest experiment folder
            output_dir = max(glob.glob("runs/detect/exp*"), key=os.path.getctime)

            # Check for output images
            annotated_image_path = os.path.join(output_dir, os.path.basename(image_path))
            if not os.path.exists(annotated_image_path):
                raise FileNotFoundError("No annotated image found after inference.")

            # Read the annotated image
            annotated_image = cv2.imread(annotated_image_path)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Extract detection labels
            label_path = os.path.join(output_dir, "labels", os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            if not os.path.exists(label_path):
                raise FileNotFoundError("No label file found for detections.")

            detections = self._parse_labels(label_path)

        except subprocess.CalledProcessError as e:
            print("Error occurred during YOLOv7 inference.")
            print("Command Output:", e.output)
            print("Error Output:", e.stderr)
            raise RuntimeError("YOLOv7 inference failed. Check logs for details.")

        finally:
            # Restore the original working directory
            os.chdir(original_cwd)

        # Pack results into a dictionary
        results = {
            "annotated_image": annotated_image,
            "detections": detections
        }

        return results

    def _parse_labels(self, label_path):
        detections = []
        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                confidence = float(parts[5])
                detections.append({
                    "class_id": class_id,
                    "bbox": bbox,  # YOLO format: [x_center, y_center, width, height]
                    "confidence": confidence
                })
        return detections

    def visualize_results(self, results):
        """
        Visualize and display annotated objects in the image.
        Args:
            results (dict): A dictionary containing the annotated image and detections.
        """
        annotated_image = results["annotated_image"]
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.title(f"YOLOv7 ({self.model_name}) Detection Results")
        plt.axis("off")
        plt.show()
