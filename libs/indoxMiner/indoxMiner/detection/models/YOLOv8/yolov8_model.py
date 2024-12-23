from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

class YOLOv8Model:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLOv8 model (you can change the model to another one as needed)
        self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        # Run object detection on the provided image
        results = self.model(image_path)
        return results

    def visualize_results(self, results):
        # Convert the image from BGR to RGB for correct display with matplotlib
        image_rgb = cv2.cvtColor(results[0].orig_img, cv2.COLOR_BGR2RGB)

        # Plot the image with bounding boxes
        plt.imshow(image_rgb)
        ax = plt.gca()

        # Draw the bounding boxes on the image
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))

        plt.axis('off')  # Remove axis
        plt.show()

