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

    def detect_objects(self, image_path, queries):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=queries, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # Image size in (height, width)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        return image, results

    def visualize_results(self, image, results, queries):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

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
            ax.text(
                xmin, ymin - 5,
                f"{queries[label]}: {score:.2f}",
                fontsize=12, color="white",
                bbox=dict(facecolor="red", alpha=0.5, edgecolor="none")
            )

        plt.axis("off")
        plt.show()
