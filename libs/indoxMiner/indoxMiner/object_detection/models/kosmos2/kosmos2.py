import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq

class Kosmos2Model:
    """
    Kosmos-2 object detection model.
    """

    def __init__(self, model_id="microsoft/kosmos-2-patch14-224", device="cuda"):
        """Initialize the Kosmos-2 model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def detect_objects(self, image_path):
        """
        Detect objects in an image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            list: Detected objects and bounding boxes.
        """
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        prompt = "<grounding> Describe the image in detail: "
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        _, entities = self.processor.post_process_generation(generated_text, cleanup_and_extract=True)

        objects = []
        for entity in entities:
            entity_name, (start, end), bbox = entity
            if start == end:
                continue
            objects.append([entity_name, bbox])

        return objects

    def visualize_results(self, image_path, objects):
        """
        Visualize detected objects in an image.

        Args:
            image_path (str): Path to the input image.
            objects (list): List of detected objects with bounding boxes.
        """
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Convert bounding boxes from normalized coordinates to pixel values
        h, w, _ = image.shape
        for entity_name, bbox_list in objects:
            bbox = bbox_list[0]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, entity_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert from BGR to RGB for display with Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        plt.imshow(image_rgb)
        plt.axis("off")  # Hide axis
        plt.show()
