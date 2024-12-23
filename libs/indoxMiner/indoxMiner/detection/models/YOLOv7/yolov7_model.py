import os
import urllib.request
import torch
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt

class YOLOv7Model:
    def __init__(self, weights_url="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt", device="cuda"):
        """
        Initialize the YOLOv7 model and download weights if necessary.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = "yolov7.pt"
        self.download_weights(weights_url)

    def download_weights(self, url):
        """
        Download the YOLOv7 weights file if it doesn't exist.
        """
        if not os.path.exists(self.weights_path):
            print(f"Downloading weights from {url}...")
            urllib.request.urlretrieve(url, self.weights_path)
            print("Download complete.")
        else:
            print("Weights file already exists.")

    def detect_objects(self, image_path, threshold=0.25):
        """
        Detect objects in an image using YOLOv7 inference.

        Args:
            image_path (str): Path to the input image.
            threshold (float): Confidence threshold for detections.

        Returns:
            str: Path to the annotated output image.
        """
        output_dir = "runs/detect/exp"
        command = f"python detect.py --weights {self.weights_path} --source {image_path} --conf-thres {threshold} --save-txt --save-conf"
        print(f"Running inference: {command}")
        os.system(command)

        # Debug: Check for output files
        output_images = glob.glob(os.path.join(output_dir, "*.jpg"))
        print(f"Output images found: {output_images}")

        if output_images:
            return output_images[0]  # Return the first found image
        else:
            raise FileNotFoundError("No output images were found after inference.")

    def visualize_results(self, output_path):
        """
        Visualize and display annotated objects in the image.

        Args:
            output_path (str): Path to the annotated output image.
        """
        # Load and display the annotated image
        annotated_image = Image.open(output_path).convert("RGB")
        plt.imshow(annotated_image)
        plt.title("Annotated Image")
        plt.axis("off")
        plt.show()