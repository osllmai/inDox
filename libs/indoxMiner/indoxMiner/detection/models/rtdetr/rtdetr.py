import torch
from PIL import Image
import supervision as sv
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import matplotlib.pyplot as plt


class RTDETR:
    """
    RT-DETR object detection model.
    """

    def __init__(self, checkpoint="PekingU/rtdetr_r50vd_coco_o365", device="cuda"):
        """Initialize the RT-DETR model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(
            self.device
        )
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)

    def detect_objects(self, image_path, threshold=0.2):
        """
        Detect objects in an image.

        Args:
            image_path (str): Path to the input image.
            threshold (float): Detection confidence threshold.

        Returns:
            dict: A dictionary containing the input image, bounding boxes, scores, and labels.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Prepare inputs for the model
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        w, h = image.size
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=threshold
        )[0]

        # Convert detections to supervision format
        detections = sv.Detections.from_transformers(results)

        # Generate labels for the detections
        labels = [
            f"{self.model.config.id2label[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Pack results into a dictionary
        packed_results = {
            "image": image,
            "detections": detections,
            "labels": labels,
        }

        return packed_results

    def visualize_results(self, packed_results):
        """
        Visualize detected objects in the image.

        Args:
            packed_results (dict): Packed results containing the input image, detections, and labels.
        """
        image = packed_results["image"]
        detections = packed_results["detections"]
        labels = packed_results["labels"]

        # Annotate the image
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        # Display the annotated image
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title("Annotated Image")
        plt.show()

