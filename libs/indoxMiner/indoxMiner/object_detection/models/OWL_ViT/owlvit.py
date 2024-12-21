import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class OWLVitModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

        # Default COCO classes as queries
        self.default_queries = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def detect_objects(self, image_path, queries=None):
        # Use default queries if none are provided
        if queries is None:
            queries = self.default_queries

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=queries, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # Image size in (height, width)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )[0]


        return image, results


    def visualize_results(self, image, results, queries=None):
        # Use default queries if none are provided
        if queries is None:
            queries = self.default_queries

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # Loop over the detection results and draw boxes
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]  # Convert to a list of rounded floats
            xmin, ymin, xmax, ymax = box

            # Draw bounding box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

            # Draw label and confidence score
            label_text = queries[label.item()] if label.item() < len(queries) else f"Class {label.item()}"
            ax.text(
                xmin, ymin - 5,
                f"{label_text}: {score:.2f}",
                fontsize=12, color="white",
                bbox=dict(facecolor="red", alpha=0.5, edgecolor="none")
            )

        plt.axis("off")
        plt.show()
