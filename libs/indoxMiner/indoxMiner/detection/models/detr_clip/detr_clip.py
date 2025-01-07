import torch
import torchvision.ops as ops
from transformers import (
    CLIPProcessor,
    CLIPModel,
    DetrImageProcessor,
    DetrForObjectDetection,
)
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


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

        self.clip_label_set = ["N/A", "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "backpack", "umbrella", "N/A", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
            "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def load_image(self, image_path):
        """
        Load and preprocess an image.
        Args:
            image_path (str): Path to the image file.
        Returns:
            PIL.Image.Image: Loaded image.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except (FileNotFoundError, UnidentifiedImageError):
            raise ValueError(f"Error: Cannot load image from path: {image_path}")

    def detect_objects(self, image=None, image_path=None, threshold=0.7, iou_threshold=0.5):
        """
        Perform object detection using DETR and classify detected regions using CLIP.
        - Applies a higher confidence threshold to reduce spurious DETR boxes.
        - Applies Non-Max Suppression (NMS) to merge overlapping boxes.
        - Uses a smaller text prompt set for CLIP to reduce confusion.
        
        Args:
            image (PIL.Image.Image, optional): Input image.
            image_path (str, optional): Path to the input image.
            threshold (float): Confidence threshold for DETR detection.
            iou_threshold (float): IOU threshold for NMS.
        Returns:
            dict: A dictionary containing image, detected objects, and additional metadata.
        """
        if image is None and image_path is not None:
            image = self.load_image(image_path)
        elif image is None:
            raise ValueError("Either an image object or image_path must be provided.")

        # Process image for DETR
        inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)

        # Run DETR model
        with torch.no_grad():
            outputs = self.detr_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        # Extract raw boxes, scores, and labels
        boxes = results["boxes"]        # (num_detections, 4)
        scores = results["scores"]      # (num_detections,)
        labels = results["labels"]      # (num_detections,)

        # Apply NMS to reduce overlapping boxes
        keep_indices = ops.nms(boxes, scores, iou_threshold=iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        clip_text_prompts = [f"a photo of a {cls}" for cls in self.clip_label_set]
        with torch.no_grad():
            clip_text_inputs = self.clip_processor(
                text=clip_text_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**clip_text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        detected_objects = []
        for (box, score_detr, label_detr) in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box.tolist()

            # Crop the detected region
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            clip_image_input = self.clip_processor(
                images=cropped_image, return_tensors="pt"
            ).to(self.device)

            # Get CLIP image features for the cropped region
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity
                similarity = torch.matmul(image_features, text_features.T)  # shape: (1, len(self.clip_label_set))
                probs = similarity.softmax(dim=-1)  # shape: (1, len(self.clip_label_set))

                # Best label index according to CLIP
                best_idx = probs.argmax(dim=1).item()
                clip_label = self.clip_label_set[best_idx]
                clip_prob = probs[0, best_idx].item()

            detected_objects.append(
                {
                    "box": (xmin, ymin, xmax, ymax),
                    "detr_label_id": label_detr.item(),
                    "detr_score": score_detr.item(),
                    "clip_label": clip_label,
                    "clip_score": clip_prob,
                }
            )

        return {
            "image": image,
            "detections": detected_objects,
            "metadata": {
                "threshold": threshold,
                "iou_threshold": iou_threshold,
                "input_size": image.size,
                "clip_label_set": self.clip_label_set,
            },
        }

    def visualize_results(self, packed_results):
        """
        Visualize bounding boxes and class labels on the image.
        Args:
            packed_results (dict): Packed results containing image and detections.
        """
        image = packed_results["image"]
        detections = packed_results["detections"]

        plt.figure(figsize=(12, 9))
        plt.imshow(image)
        ax = plt.gca()

        for detection in detections:
            xmin, ymin, xmax, ymax = detection["box"]
            clip_label = detection["clip_label"]
            clip_prob = detection["clip_score"]

            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )
            ax.text(
                xmin,
                ymin,
                f"{clip_label}: {clip_prob:.3f}",
                color="yellow",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.5),
            )

        plt.axis("off")
        plt.title("DETR + CLIP Object Detection (with NMS & limited classes)")
        plt.show()
