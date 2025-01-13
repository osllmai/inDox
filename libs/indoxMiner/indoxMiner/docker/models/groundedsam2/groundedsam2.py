import os
import cv2
import json
import torch
import numpy as np
import subprocess
import supervision as sv
import sys
import requests
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import matplotlib.pyplot as plt


class GroundedSAM2:
    def __init__(
        self,
        config_name_or_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        weights_name_or_path=None,
        grounding_model="IDEA-Research/grounding-dino-tiny",
        device="cuda",
        default_text_prompt=None,
    ):
        """
        Initializes the GroundedSAM2 object segmentation model.

        Args:
            config_name_or_path (str, optional): Path to the config file.
            weights_name_or_path (str, optional): Path to the model weights file.
            grounding_model (str): Hugging Face model ID for GroundingDINO.
            device (str): Device to run the model on ('cuda' or 'cpu').
            default_text_prompt (str, optional): Default text prompt for object detection.
        """
        self.device = device
        self.grounding_model_id = grounding_model

        # Set default text prompt to include COCO classes if not provided
        self.default_text_prompt = default_text_prompt or (
            "person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. "
            "parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. "
            "handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. "
            "surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. "
            "broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. TV. laptop. "
            "mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. "
            "teddy bear. hair drier. toothbrush."
        )

        # self._install_grounded_sam2()

        # Automatically download config and weight files if necessary
        self.config_path = config_name_or_path
        self.weights_path = (
            self._get_file(weights_name_or_path, "weights")
            if weights_name_or_path
            else self._get_file("sam2.1_hiera_large.pt", "weights")
        )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Build SAM2 model and predictor
        self.sam2_model = build_sam2(
            self.config_path, self.weights_path, device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Load GroundingDINO model
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_id
        ).to(self.device)

    # def _install_grounded_sam2(self):
    #     """
    #     Installs the Grounded SAM2 library if not already installed.
    #     """
    #     try:
    #         import sam2
    #     except ImportError:
    #         print("Grounded_SAM2 not found. Installing...")
    #         subprocess.check_call(
    #             [
    #                 sys.executable,
    #                 "-m",
    #                 "pip",
    #                 "install",
    #                 "git+https://github.com/IDEA-Research/Grounded-SAM-2.git",
    #             ]
    #         )

    def _get_file(self, name_or_path, file_type):
        """
        Downloads the weights file if only a name is provided.

        Args:
            name_or_path (str): Name or path to the file.
            file_type (str): Type of file ('weights').

        Returns:
            str: Path to the file.
        """
        if os.path.exists(name_or_path):
            return name_or_path

        # Define URL for known file
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

        file_name = os.path.basename(url)

        # Download the file
        print(f"Downloading {file_type} file: {file_name}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"{file_type.capitalize()} file downloaded: {file_name}")
            return file_name
        else:
            raise RuntimeError(
                f"Failed to download {file_type} file from {url} (status code: {response.status_code})"
            )

    def detect_objects(self, image_path, text_prompt=None):
        """
        Detect objects in the image using Grounded SAM2.

        Args:
            image_path (str): Path to the input image.
            text_prompt (str, optional): Text prompt for object detection. If None, uses default_text_prompt.

        Returns:
            dict: A dictionary containing the input image, detections, and labels.
        """
        image = Image.open(image_path).convert("RGB")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

        self.sam2_predictor.set_image(np.array(image))
        text_prompt = text_prompt or self.default_text_prompt
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )

        # Perform inference
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
        )

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        # Return packed results
        return {
            "image_bgr": img_bgr,
            "detections": detections,
            "labels": labels,
        }

    def visualize_results(self, packed_results):
        """
        Visualize detected objects on the image.

        Args:
            packed_results (dict): Packed results containing image, detections, and labels.

        Displays the annotated image.
        """
        image_bgr = packed_results["image_bgr"]
        detections = packed_results["detections"]
        labels = packed_results["labels"]

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image_bgr.copy(), detections=detections
        )

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
