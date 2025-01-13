# import torch
# from transformers import (
#     CLIPProcessor,
#     CLIPModel,
#     DetrImageProcessor,
#     DetrForObjectDetection,
# )
# from PIL import Image, UnidentifiedImageError
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np


# class DETRCLIP:
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch32",
#         detr_model_name="facebook/detr-resnet-50",
#         device="cuda",
#     ):
#         """
#         Initialize the DETR and CLIP models.
#         Args:
#             clip_model_name (str): Pretrained CLIP model name.
#             detr_model_name (str): Pretrained DETR model name.
#             device (str): Device to run the models ("cuda" or "cpu").
#         """
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")

#         # Initialize CLIP model and processor
#         self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
#         self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

#         # Initialize DETR model and processor
#         self.detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(
#             self.device
#         )
#         self.detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)
#         self.detr_model.eval()

#         # COCO class labels (91 classes)
#         self.coco_classes = [
#             "N/A",
#             "person",
#             "bicycle",
#             "car",
#             "motorcycle",
#             "airplane",
#             "bus",
#             "train",
#             "truck",
#             "boat",
#             "traffic light",
#             "fire hydrant",
#             "N/A",
#             "stop sign",
#             "parking meter",
#             "bench",
#             "bird",
#             "cat",
#             "dog",
#             "horse",
#             "sheep",
#             "cow",
#             "elephant",
#             "bear",
#             "zebra",
#             "giraffe",
#             "N/A",
#             "backpack",
#             "umbrella",
#             "N/A",
#             "handbag",
#             "tie",
#             "suitcase",
#             "frisbee",
#             "skis",
#             "snowboard",
#             "sports ball",
#             "kite",
#             "baseball bat",
#             "baseball glove",
#             "skateboard",
#             "surfboard",
#             "tennis racket",
#             "bottle",
#             "N/A",
#             "wine glass",
#             "cup",
#             "fork",
#             "knife",
#             "spoon",
#             "bowl",
#             "banana",
#             "apple",
#             "sandwich",
#             "orange",
#             "broccoli",
#             "carrot",
#             "hot dog",
#             "pizza",
#             "donut",
#             "cake",
#             "chair",
#             "couch",
#             "potted plant",
#             "bed",
#             "N/A",
#             "dining table",
#             "N/A",
#             "toilet",
#             "N/A",
#             "tv",
#             "laptop",
#             "mouse",
#             "remote",
#             "keyboard",
#             "cell phone",
#             "microwave",
#             "oven",
#             "toaster",
#             "sink",
#             "refrigerator",
#             "book",
#             "clock",
#             "vase",
#             "scissors",
#             "teddy bear",
#             "hair drier",
#             "toothbrush",
#         ]

#     def detect_objects(self, image=None, image_path=None, threshold=0.5):
#         """
#         Perform object detection using DETR and classify detected regions using CLIP.
#         Args:
#             image (PIL.Image.Image, optional): Input image.
#             image_path (str, optional): Path to the input image.
#             threshold (float): Confidence threshold for detections.
#         Returns:
#             list: Detected objects with bounding boxes, class labels, and probabilities.
#             PIL.Image.Image: Loaded image (if image_path was provided).
#         """
#         if image is None and image_path is not None:
#             try:
#                 image = Image.open(image_path).convert("RGB")
#             except (FileNotFoundError, UnidentifiedImageError):
#                 raise ValueError(f"Error: Cannot load image from path: {image_path}")
#         elif image is None:
#             raise ValueError("Either an image object or image_path must be provided.")

#         # Process image for DETR
#         inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)

#         # Run DETR model
#         with torch.no_grad():
#             outputs = self.detr_model(**inputs)

#         target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
#         results = self.detr_processor.post_process_object_detection(
#             outputs, target_sizes=target_sizes, threshold=threshold
#         )[0]

#         # Preprocess COCO classes for CLIP
#         clip_text_inputs = self.clip_processor(
#             text=[f"a photo of a {cls}" for cls in self.coco_classes],
#             return_tensors="pt",
#             padding=True,
#         ).to(self.device)

#         with torch.no_grad():
#             text_features = self.clip_model.get_text_features(**clip_text_inputs)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#         detected_objects = []
#         for score, label, box in zip(
#             results["scores"], results["labels"], results["boxes"]
#         ):
#             xmin, ymin, xmax, ymax = box.tolist()

#             # Crop the detected region
#             cropped_image = image.crop((xmin, ymin, xmax, ymax))
#             clip_image_input = self.clip_processor(
#                 images=cropped_image, return_tensors="pt"
#             ).to(self.device)

#             # Get CLIP image features for the cropped region
#             with torch.no_grad():
#                 image_features = self.clip_model.get_image_features(**clip_image_input)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)

#                 # Compute cosine similarity
#                 similarity = torch.matmul(image_features, text_features.T)
#                 probs = similarity.softmax(dim=-1)
#                 detected_class_idx = probs.argmax().item()
#                 detected_class = self.coco_classes[detected_class_idx]
#                 detected_prob = probs[0, detected_class_idx].item()

#             detected_objects.append(
#                 {
#                     "box": (xmin, ymin, xmax, ymax),
#                     "label": detected_class,
#                     "score": detected_prob,
#                 }
#             )

#         return detected_objects, image

#     def visualize_results(self, outputs):
#         """
#         Visualize detection results.
#         Args:
#             outputs (dict): Dictionary containing 'instances' and 'orig_img'
#         Returns:
#             numpy.ndarray: Visualized image with detections
#         """
#         image = outputs["orig_img"]
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         instances = outputs["instances"]
#         class_names = instances.get("class_names", self.coco_classes)

#         # Create figure and axis
#         fig, ax = plt.subplots(1, figsize=(12, 8))
#         ax.imshow(image_rgb)

#         # Draw boxes and labels
#         for box, score, label_idx in zip(
#             instances["pred_boxes"], instances["scores"], instances["pred_classes"]
#         ):
#             xmin, ymin, xmax, ymax = box.tolist() if torch.is_tensor(box) else box
#             label = class_names[int(label_idx)]

#             # Draw bounding box
#             rect = plt.Rectangle(
#                 (xmin, ymin),
#                 xmax - xmin,
#                 ymax - ymin,
#                 fill=False,
#                 color="red",
#                 linewidth=2,
#             )
#             ax.add_patch(rect)

#             # Add label
#             ax.text(
#                 xmin,
#                 ymin - 10,
#                 f"{label}: {score:.2f}",
#                 color="red",
#                 fontsize=10,
#                 bbox=dict(facecolor="white", alpha=0.7),
#             )

#         ax.axis("off")

#         # Convert matplotlib figure to numpy array
#         fig.canvas.draw()
#         img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         plt.close(fig)

#         return img_array

#     def save_results(self, image, output_path="output.jpg"):
#         """Saves the output image with drawn bounding boxes."""
#         cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#         print(f"Results saved to {output_path}")


import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    DetrImageProcessor,
    DetrForObjectDetection,
)
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import cv2


class DETRCLIP:
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        detr_model_name="facebook/detr-resnet-50",
        device="cuda",
    ):
        """
        Initialize the DETR and CLIP models.
        Args:
            clip_model_name (str): Pretrained CLIP model name.
            detr_model_name (str): Pretrained DETR model name.
            device (str): Device to run the models ("cuda" or "cpu").
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Initialize DETR model and processor
        self.detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(
            self.device
        )
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)
        self.detr_model.eval()

        # COCO class labels
        self.coco_classes = [
            "N/A",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "N/A",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "N/A",
            "backpack",
            "umbrella",
            "N/A",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "N/A",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "N/A",
            "dining table",
            "N/A",
            "toilet",
            "N/A",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def detect_objects(self, image_path, threshold=0.5):
        """
        Perform object detection using DETR and classify detected regions using CLIP.
        Args:
            image_path (str): Path to the input image.
            threshold (float): Confidence threshold for detections.
        Returns:
            dict: Contains 'instances' with detection results and 'orig_img'
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error: Cannot load image from path: {image_path}")

            # Convert to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Process image for DETR
            inputs = self.detr_processor(images=pil_image, return_tensors="pt").to(
                self.device
            )

            # Run DETR model
            with torch.no_grad():
                outputs = self.detr_model(**inputs)

            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

            # Preprocess COCO classes for CLIP
            clip_text_inputs = self.clip_processor(
                text=[f"a photo of a {cls}" for cls in self.coco_classes],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**clip_text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Process each detection with CLIP
            boxes = []
            scores = []
            labels = []

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                xmin, ymin, xmax, ymax = box.tolist()
                boxes.append([xmin, ymin, xmax, ymax])

                # Crop the detected region
                cropped_region = pil_image.crop((xmin, ymin, xmax, ymax))
                clip_image_input = self.clip_processor(
                    images=cropped_region, return_tensors="pt"
                ).to(self.device)

                # Get CLIP features and classification
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(
                        **clip_image_input
                    )
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = torch.matmul(image_features, text_features.T)
                    probs = similarity.softmax(dim=-1)
                    detected_class_idx = probs.argmax().item()
                    detected_prob = probs[0, detected_class_idx].item()

                scores.append(detected_prob)
                labels.append(detected_class_idx)

            # Create instances dictionary
            instances = {
                "pred_boxes": torch.tensor(boxes),
                "scores": torch.tensor(scores),
                "pred_classes": torch.tensor(labels),
                "class_names": self.coco_classes,  # Include class names for visualization
            }

            return {"instances": instances, "orig_img": image}

        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def visualize_results(self, outputs):
        """
        Visualize detection results.
        Args:
            outputs (dict): Dictionary containing 'instances' and 'orig_img'
        Returns:
            numpy.ndarray: Visualized image with detections
        """
        image = outputs["orig_img"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances = outputs["instances"]
        class_names = instances.get("class_names", self.coco_classes)

        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)

        # Draw boxes and labels
        for box, score, label_idx in zip(
            instances["pred_boxes"], instances["scores"], instances["pred_classes"]
        ):
            xmin, ymin, xmax, ymax = box.tolist() if torch.is_tensor(box) else box
            label = class_names[int(label_idx)]

            # Draw bounding box
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                xmin,
                ymin - 10,
                f"{label}: {score:.2f}",
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
