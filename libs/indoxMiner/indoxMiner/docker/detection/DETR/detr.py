import torch
from PIL import Image, UnidentifiedImageError
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DETR:
    def __init__(self, checkpoint="facebook/detr-resnet-50", device="cuda"):
        """
        Initialize the DETR model and processor.
        Args:
            checkpoint (str): Pretrained model checkpoint.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DetrForObjectDetection.from_pretrained(checkpoint).to(self.device)
        self.processor = DetrImageProcessor.from_pretrained(checkpoint)

    def detect_objects(self, image_path, resize_width=800, score_threshold=0.7):
        """
        Perform object detection on an image given its path.
        Args:
            image_path (str): Path to the image file.
            resize_width (int): Width to resize image while maintaining aspect ratio.
            score_threshold (float): Minimum confidence score for detections.
        Returns:
            dict: Filtered detections with bounding boxes, labels, and scores.
        """
        try:
            # Load and resize image
            image = Image.open(image_path).convert("RGB")
            aspect_ratio = image.height / image.width
            new_height = int(resize_width * aspect_ratio)
            image = image.resize((resize_width, new_height))

            w, h = image.size
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process the results to COCO format
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)]
            )[0]

            # Apply confidence threshold
            indices = results["scores"] > score_threshold
            return {
                "boxes": results["boxes"][indices],
                "scores": results["scores"][indices],
                "labels": results["labels"][indices],
                "image": image,
            }
        except (FileNotFoundError, UnidentifiedImageError) as e:
            raise ValueError(f"Error loading image: {e}")

    def visualize_results(self, results):
        """
        Visualize detected objects on the image.
        Args:
            results (dict): Results from the detect_objects method.
        """
        image = results["image"]
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        ax = plt.gca()

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 10,
                f"{self.model.config.id2label[label.item()]}: {score:.2f}",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        plt.axis("off")
        plt.title("Bounding Box Results")
        plt.show()
