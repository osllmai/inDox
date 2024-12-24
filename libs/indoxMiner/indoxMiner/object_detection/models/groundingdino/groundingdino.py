import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow  # For Colab compatibility
import subprocess
import sys
import requests


class GroundingDINOModel:
    def __init__(
        self,
        config_name_or_path,
        weights_name_or_path,
        device="cuda",
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        """
        Initializes the Grounding DINO object detection model.

        Args:
            config_name_or_path (str): Name or path to the config file. If name is provided, it will be downloaded.
            weights_name_or_path (str): Name or path to the model weights file. If name is provided, it will be downloaded.
            device (str): Device to run the model on ('cuda' or 'cpu').
            box_threshold (float): Threshold for box confidence.
            text_threshold (float): Threshold for text confidence.
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Default COCO classes as text prompt
        self.text_prompt = (
            "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, "
            "fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, "
            "bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
            "sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, "
            "wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
            "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, "
            "mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, "
            "vase, scissors, teddy bear, hair drier, toothbrush"
        )

        # Ensure Grounding DINO is installed
        self._install_groundingdino()

        # Automatically download config and weight files if necessary
        self.config_path = self._get_file(config_name_or_path, "config")
        self.weights_path = self._get_file(weights_name_or_path, "weights")

        # Import Grounding DINO utilities
        from groundingdino.util.inference import load_model

        # Load the model
        self.model = load_model(self.config_path, self.weights_path)
        self.model.to(self.device).eval()

    def _install_groundingdino(self):
        """
        Installs the groundingdino package if not already installed.
        """
        try:
            import groundingdino
        except ImportError:
            print("Grounding DINO not found. Installing...")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/IDEA-Research/GroundingDINO.git",
                ]
            )

    def _get_file(self, name_or_path, file_type):
        """
        Downloads the config or weights file if only a name is provided.

        Args:
            name_or_path (str): Name or path to the file.
            file_type (str): Type of file ('config' or 'weights').

        Returns:
            str: Path to the file.
        """
        if os.path.exists(name_or_path):
            return name_or_path

        # Define URLs for known files
        urls = {
            "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        }

        if file_type not in urls:
            raise ValueError(f"Unknown file type: {file_type}")

        file_url = urls[file_type]
        file_name = name_or_path if name_or_path else os.path.basename(file_url)

        # Download the file
        print(f"Downloading {file_type} file: {file_name}...")
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"{file_type.capitalize()} file downloaded: {file_name}")
            return file_name
        else:
            raise RuntimeError(
                f"Failed to download {file_type} file from {file_url} (status code: {response.status_code})"
            )

    def detect_objects(self, image_path):
        """
        Detect objects in the image using the default text prompt.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Dictionary containing the image and detection results, including boxes, logits, and phrases.
        """
        from groundingdino.util.inference import load_image, predict

        image_source, image = load_image(image_path)

        # Perform object detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        return {
            "image_source": image_source,
            "boxes": boxes,
            "logits": logits,
            "phrases": phrases,
        }

    def visualize_results(self, result):
        """
        Visualize the detection results on the image.

        Args:
            result (dict): Dictionary containing the image and detection results, including boxes, logits, and phrases.
        """
        from groundingdino.util.inference import annotate

        image_source = result["image_source"]
        boxes = result["boxes"]
        logits = result["logits"]
        phrases = result["phrases"]

        # Annotate the image with the detected boxes and phrases
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )

        # Display the annotated image using cv2_imshow in Colab
        cv2_imshow(annotated_frame)
