import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt


class OWLv2:
    """
    OWL-V2 object detection model with support for default COCO queries.
    """

    def __init__(self, queries=None):
        """
        Initialize the OWL-V2 model with default or user-provided queries.

        Args:
            queries (list, optional): List of object queries (e.g., ["cat", "dog"]).
                                      If None, defaults to COCO classes as queries.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

        # Default COCO class queries
        self.coco_classes = [
            "a person", "a bicycle", "a car", "a motorcycle", "an airplane", "a bus", 
            "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", 
            "a stop sign", "a parking meter", "a bench", "a bird", "a cat", "a dog", 
            "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", 
            "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", 
            "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", 
            "a kite", "a baseball bat", "a baseball glove", "a skateboard", 
            "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", 
            "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", 
            "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", 
            "a donut", "a cake", "a chair", "a couch", "a potted plant", "a bed", 
            "a dining table", "a toilet", "a TV", "a laptop", "a mouse", "a remote", 
            "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", 
            "a sink", "a refrigerator", "a book", "a clock", "a vase", "scissors", 
            "a teddy bear", "a hair dryer", "a toothbrush"
        ]

        # Use default COCO classes unless custom queries are provided
        self.queries = [self.coco_classes] if queries is None else queries

    def detect_objects(self, image_path):
        """
        Detect objects in an image using OWL-V2.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the processed image, bounding boxes, scores, and labels.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=self.queries, images=image, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.2
        )[0]

        # Pack all outputs into a dictionary
        packed_results = {
            "image": image,
            "boxes": results["boxes"].cpu().tolist(),
            "scores": results["scores"].cpu().tolist(),
            "labels": [self.queries[0][label] for label in results["labels"].cpu().tolist()],
        }

        return packed_results

    def visualize_results(self, packed_results):
        """
        Visualize detected objects in an image.

        Args:
            packed_results (dict): A dictionary containing detection results.
        """
        image = packed_results["image"]
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # Default font; customize if needed

        for box, score, label in zip(
            packed_results["boxes"], packed_results["scores"], packed_results["labels"]
        ):
            x1, y1, x2, y2 = [round(coord, 2) for coord in box]

            # Draw bounding box
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)

            # Draw label and score
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)  # Use `textbbox` to get the bounding box
            text_background = [(text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])]
            draw.rectangle(text_background, fill="red")
            draw.text((x1, y1), text, fill="white", font=font)

        # Display the image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
