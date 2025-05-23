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

    def detect_objects(self, image_path, threshold=0.1):
        """
        Detect objects in an image.

        Args:
            image_path (str): Path to the input image.
            threshold (float): Detection confidence threshold.

        Returns:
            PIL.Image.Image: The input image.
            dict: Detected objects with bounding boxes and scores.
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
        )
        return image, results[0]

    def visualize_results(self, image, results):
        """
        Visualize detected objects in the image.

        Args:
            image (PIL.Image.Image): The input image.
            results (dict): The results from the detect_objects method.
        """
        # Convert detections from the model's output
        detections = sv.Detections.from_transformers(results)

        # Generate labels for the detections
        labels = [
            f"{self.model.config.id2label[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

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


# import numpy as np
# import torch
# from PIL import Image
# import numpy as np
# from transformers import AutoImageProcessor, RTDetrForObjectDetection
# import matplotlib.pyplot as plt
# import supervision as sv


# class RTDETR:
#     """
#     RT-DETR object detection model for FastAPI integration.
#     """

#     def __init__(self, device="cuda"):
#         """
#         Initialize the RT-DETR model.

#         Args:
#             device (str): Device to run model on ('cuda' or 'cpu')
#         """
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
#         checkpoint = "PekingU/rtdetr_r50vd_coco_o365"
#         self.model = RTDetrForObjectDetection.from_pretrained(checkpoint).to(
#             self.device
#         )
#         self.processor = AutoImageProcessor.from_pretrained(checkpoint)

#     def detect_objects(self, image_path, threshold=0.9):
#         """
#         Detect objects in an image.

#         Args:
#             image_path (str): Path to the image file
#             threshold (float): Detection confidence threshold

#         Returns:
#             dict: Dictionary containing 'boxes', 'scores', and 'labels' as numpy arrays
#         """
#         # Load and preprocess image
#         image = Image.open(image_path).convert("RGB")
#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)

#         # Perform inference
#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         # Post-process outputs
#         target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
#         results = self.processor.post_process_object_detection(
#             outputs, threshold=threshold, target_sizes=target_sizes
#         )[0]

#         # Convert to numpy arrays for API response
#         return {
#             "boxes": np.array(results["boxes"]),
#             "scores": np.array(results["scores"]),
#             "labels": np.array(results["labels"]),
#             "image": image,  # Keep the image for visualization
#         }

#     def visualize_results(self, image, results):
#         """
#         Visualize detected objects in the image.

#         Args:
#             image (PIL.Image.Image): The input image.
#             results (dict): The results from the detect_objects method.
#         """
#         # Convert detections from the model's output
#         detections = sv.Detections.from_transformers(results)

#         # Generate labels for the detections
#         labels = [
#             f"{self.model.config.id2label[class_id]} {confidence:.2f}"
#             for class_id, confidence in zip(detections.class_id, detections.confidence)
#         ]

#         # Annotate the image
#         box_annotator = sv.BoxAnnotator()
#         annotated_image = box_annotator.annotate(scene=image, detections=detections)

#         label_annotator = sv.LabelAnnotator()
#         annotated_image = label_annotator.annotate(
#             scene=annotated_image, detections=detections, labels=labels
#         )

#         # Display the annotated image
#         plt.imshow(annotated_image)
#         plt.axis("off")
#         plt.title("Annotated Image - RTDETR Model")
#         plt.show()
