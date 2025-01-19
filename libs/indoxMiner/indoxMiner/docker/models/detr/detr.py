import torch
from PIL import Image, UnidentifiedImageError
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2


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
            dict: Detection results with 'instances' key containing boxes, scores, and labels
        """
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Resize while maintaining aspect ratio
            aspect_ratio = pil_image.height / pil_image.width
            new_height = int(resize_width * aspect_ratio)
            pil_image = pil_image.resize((resize_width, new_height))

            # Prepare inputs
            inputs = self.processor(images=pil_image, return_tensors="pt").to(
                self.device
            )

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=[(pil_image.height, pil_image.width)]
            )[0]

            # Filter by confidence
            keep_idx = results["scores"] > score_threshold
            boxes = results["boxes"][keep_idx]
            scores = results["scores"][keep_idx]
            labels = results["labels"][keep_idx]

            # Scale boxes back to original image size
            scale_x = image.shape[1] / pil_image.width
            scale_y = image.shape[0] / pil_image.height
            scaled_boxes = boxes.clone()
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 3] *= scale_y

            # Create instances dict
            instances = {
                "pred_boxes": scaled_boxes.cpu(),
                "scores": scores.cpu(),
                "pred_classes": labels.cpu(),
            }

            return {"instances": instances, "orig_img": image}

        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def visualize_results(self, outputs):
        """
        Visualize detection results.
        Args:
            outputs (dict): Dictionary containing 'instances' and 'orig_img'.
        Returns:
            numpy.ndarray: Visualized image with detections.
        """
        image = outputs["orig_img"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances = outputs["instances"]

        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)

        # Draw boxes and labels
        for box, score, label in zip(
            instances["pred_boxes"], instances["scores"], instances["pred_classes"]
        ):
            x_min, y_min, x_max, y_max = box.numpy()
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Get label name if available
            if hasattr(self.model.config, "id2label"):
                label_name = self.model.config.id2label[label.item()]
            else:
                label_name = f"Class {label.item()}"

            ax.text(
                x_min,
                y_min - 10,
                f"{label_name}: {score:.2f}",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        ax.axis("off")

        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img_array

    def save_results(self, image, output_path="output.jpg"):
        """Saves the output image with drawn bounding boxes."""
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Results saved to {output_path}")
