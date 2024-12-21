import torch
from transformers import CLIPProcessor, CLIPModel, DetrImageProcessor, DetrForObjectDetection
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

class DETRCLIPModel:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", detr_model_name="facebook/detr-resnet-50", device="cuda"):
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
        self.detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(self.device)
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)
        self.detr_model.eval()

        # COCO class labels (91 classes)
        self.coco_classes = [
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

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

    def detect_objects(self, image, threshold=0.5):
        """
        Perform object detection using DETR and classify detected regions using CLIP.
        Args:
            image (PIL.Image.Image): Input image.
            threshold (float): Confidence threshold for detections.
        Returns:
            list: Detected objects with bounding boxes, class labels, and probabilities.
        """
        # Process image for DETR
        inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)

        # Run DETR model
        with torch.no_grad():
            outputs = self.detr_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        # Preprocess COCO classes for CLIP
        clip_text_inputs = self.clip_processor(
            text=[f"a photo of a {cls}" for cls in self.coco_classes],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**clip_text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()

            # Crop the detected region
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            clip_image_input = self.clip_processor(images=cropped_image, return_tensors="pt").to(self.device)

            # Get CLIP image features for the cropped region
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity
                similarity = torch.matmul(image_features, text_features.T)
                probs = similarity.softmax(dim=-1)
                detected_class_idx = probs.argmax().item()
                detected_class = self.coco_classes[detected_class_idx]
                detected_prob = probs[0, detected_class_idx].item()

            detected_objects.append({
                "box": (xmin, ymin, xmax, ymax),
                "label": detected_class,
                "score": detected_prob
            })

        return detected_objects

    def visualize_results(self, image, detections):
        """
        Visualize bounding boxes and class labels on the image.
        Args:
            image (PIL.Image.Image): Original image.
            detections (list): Detected objects with bounding boxes, labels, and scores.
        """
        plt.figure(figsize=(12, 9))
        plt.imshow(image)
        ax = plt.gca()

        for detection in detections:
            xmin, ymin, xmax, ymax = detection["box"]
            label = detection["label"]
            score = detection["score"]

            # Draw bounding box
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=2))
            ax.text(xmin, ymin, f"{label}: {score:.3f}", color="yellow", fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

        plt.axis("off")
        plt.title("DETR + CLIP Object Detection")
        plt.show()


