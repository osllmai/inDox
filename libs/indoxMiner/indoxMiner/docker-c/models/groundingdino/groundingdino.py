import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDINO:
    def __init__(
        self,
        model_name_or_path="IDEA-Research/grounding-dino-tiny",
        device="cuda",
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        """
        Initializes the Grounding DINO object detection model.

        Args:
            model_name_or_path (str): Hugging Face model name or path for GroundingDINO.
            device (str): Device to run the model on ('cuda' or 'cpu').
            box_threshold (float): Threshold for box confidence.
            text_threshold (float): Threshold for text confidence.
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Default COCO classes as text prompt
        self.text_prompt = (
            "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . "
            "fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . "
            "bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . "
            "sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . "
            "wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . "
            "hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . "
            "mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . "
            "vase . scissors . teddy bear . hair drier . toothbrush ."
        )

        # Load model and processor from Hugging Face
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name_or_path
        ).to(self.device)

    def detect_objects(self, image_path):
        """
        Detect objects in the image using the default text prompt.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Dictionary containing the image and detection results, including boxes, scores, and labels.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess image and text prompt
        inputs = self.processor(
            images=image, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],  # اضافه کردن input_ids
            target_sizes=[image.size[::-1]],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        return {
            "image": image,
            "boxes": results[0]["boxes"].cpu().numpy(),
            "scores": results[0]["scores"].cpu().numpy(),
            "labels": results[0]["labels"],
        }

    def visualize_results(self, result):
        """
        Visualize detection results.

        Args:
            result (dict): Contains image and detection results (boxes, scores, labels).
        """
        import matplotlib.patches as patches

        image = result["image"]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # Plot image
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)

        # Add boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min

            # Add rectangle
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x_min,
                y_min - 10,
                f"{label} {score:.2f}",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

        plt.axis("off")
        plt.show()
