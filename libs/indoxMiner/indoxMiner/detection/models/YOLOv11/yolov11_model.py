import os
import subprocess
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class YOLOv11Model:
    def __init__(self, model_path="yolo11n.pt", device="cuda"):
        """
        Initialize YOLOv11 model.
        Args:
            model_path (str): Path to YOLOv11 model weights (e.g., 'yolo11n.pt').
            device (str): Device to use ("cuda" or "cpu").
        """
        self.model_path = model_path
        self.device = device

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
        latest_image = max(glob.glob(f"{result_dir}/*.jpg"), key=os.path.getctime)
        return latest_image



    def visualize_results(self, result_image_path):
        """
        Visualize the YOLOv11 predictions.
        Args:
            result_image_path (str): Path to the output image saved by YOLOv11.
        """
        img = mpimg.imread(result_image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("YOLOv11 Object Detection")
        plt.show()
