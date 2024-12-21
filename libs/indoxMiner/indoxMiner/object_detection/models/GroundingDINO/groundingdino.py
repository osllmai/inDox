import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from google.colab.patches import cv2_imshow  # For Colab compatibility

class GroundingDINOObjectDetector:
    def __init__(self, config_path, model_weights_path, device='cuda', box_threshold=0.35, text_threshold=0.25):
        """
        Initializes the Grounding DINO object detection model.

        Args:
            config_path (str): Path to the config file.
            model_weights_path (str): Path to the model weights file.
            device (str): Device to run the model on ('cuda' or 'cpu').
            box_threshold (float): Threshold for box confidence.
            text_threshold (float): Threshold for text confidence.
        """
        self.device = device
        self.model = load_model(config_path, model_weights_path)
        self.model.to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect_objects(self, image_path, text_prompt):
        """
        Detect objects in the image using the given text prompt.

        Args:
            image_path (str): Path to the input image.
            text_prompt (str): The prompt to guide object detection (e.g., "chair, person, dog").

        Returns:
            result (dict): Dictionary containing the image and detection results, including boxes, logits, and phrases.
        """
        image_source, image = load_image(image_path)

        # Perform object detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # Return result as a dictionary
        return {
            'image_source': image_source,
            'boxes': boxes,
            'logits': logits,
            'phrases': phrases
        }

    def visualize_results(self, result):
        """
        Visualize the detection results on the image.

        Args:
            result (dict): Dictionary containing the image and detection results, including boxes, logits, and phrases.
        """
        image_source = result['image_source']
        boxes = result['boxes']
        logits = result['logits']
        phrases = result['phrases']

        # Annotate the image with the detected boxes and phrases
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        # Display the annotated image using cv2_imshow in Colab
        cv2_imshow(annotated_frame)