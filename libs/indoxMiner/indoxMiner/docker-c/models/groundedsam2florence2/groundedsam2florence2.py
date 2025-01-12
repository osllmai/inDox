import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import subprocess
import sys
import requests


class GroundedSAM2Florence2:
    def __init__(
        self,
        config_name_or_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        weights_name_or_path=None,
        florence2_model_id="microsoft/Florence-2-large",
        device="cuda",
    ):
        """
        Initializes the GroundedSAM2Florence2 pipeline.

        Args:
            config_name_or_path (str): Path to the SAM2 config file. Defaults to 'configs/sam2.1/sam2.1_hiera_l.yaml'.
            weights_name_or_path (str, optional): Path to the SAM2 model weights file. If None, it will be downloaded.
            florence2_model_id (str): Model ID for Florence-2 from Hugging Face.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.task_prompt = {
            "object_detection": "<OD>",
            "dense_region_caption": "<DENSE_REGION_CAPTION>",
            "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        }

        # Ensure Grounded-SAM2 is installed
        self._install_grounded_sam2()

        # Download weights if not provided
        self.config_path = config_name_or_path
        self.weights_path = (
            self._get_file(weights_name_or_path, "weights")
            if weights_name_or_path
            else self._get_file("sam2.1_hiera_large.pt", "weights")
        )

        # Load Florence-2
        self.florence2_model = (
            AutoModelForCausalLM.from_pretrained(
                florence2_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .eval()
            .to(device)
        )
        self.florence2_processor = AutoProcessor.from_pretrained(
            florence2_model_id, trust_remote_code=True
        )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Load SAM2
        self.sam2_model = build_sam2(
            config_name_or_path, self.weights_path, device=device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def _install_grounded_sam2(self):
        """
        Installs the Grounded SAM2 library if not already installed.
        """
        try:
            import sam2
        except ImportError:
            print("SAM2 not found. Installing...")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/IDEA-Research/Grounded-SAM-2.git",
                ]
            )

    def _get_file(self, name_or_path, file_type):
        """
        Downloads the config or weights file if only a name is provided.

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

    def run_florence2(self, task_prompt, text_input, image):
        """
        Runs Florence-2 for the given task.

        Args:
            task_prompt (str): Task-specific prompt for Florence-2.
            text_input (str): Additional text input for the task.
            image (PIL.Image): Input image.

        Returns:
            dict: Parsed results from Florence-2.
        """
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.florence2_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(
            torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            generated_ids = self.florence2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = self.florence2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.florence2_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=image.size
        )
        return parsed_answer

    def detect_objects(self, image_path):
        """
        Runs object detection and segmentation pipeline.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the input image (BGR format), detections, and labels.
        """
        image = Image.open(image_path).convert("RGB")
        results = self.run_florence2(self.task_prompt["object_detection"], None, image)

        input_boxes = np.array(results[self.task_prompt["object_detection"]]["bboxes"])
        class_names = results[self.task_prompt["object_detection"]]["labels"]
        class_ids = np.arange(len(class_names))

        # Predict masks with SAM2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        img_bgr = cv2.imread(image_path)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
        )

        # Return packed results
        return {
            "image_bgr": img_bgr,
            "detections": detections,
            "labels": class_names,
        }

    def visualize_results(self, packed_results):
        """
        Visualize detected objects and masks on the image.

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


# Task Prompts Explanation:
# - "<OD>": Object Detection - Detects objects and their bounding boxes.
# - "<DENSE_REGION_CAPTION>": Dense Region Captioning - Generates captions for dense regions in the image.
# - "<CAPTION_TO_PHRASE_GROUNDING>": Phrase Grounding - Maps phrases in a caption to regions in the image.
