import os
import sys
import subprocess
import requests
import cv2
import matplotlib.pyplot as plt


class YOLOv8:
    def __init__(self, weight_name="yolov8n.pt"):
        """
        Initialize YOLOv8 model.
        Args:
            weight_name (str): Name or path to the YOLOv8 model weights.
        """
        # Ensure ultralytics is installed
        self._install_ultralytics()

        # Import YOLO after ensuring ultralytics is installed
        from ultralytics import YOLO

        self.model_path = self._get_model_weights(weight_name)
        self.model = YOLO(self.model_path)

    def _install_ultralytics(self):
        """
        Installs the ultralytics library if not already installed.
        """
        try:
            import ultralytics
        except ImportError:
            print("Ultralytics library not found. Installing...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "ultralytics"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("Ultralytics installed successfully.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install ultralytics: {e}")

    def _get_model_weights(self, weight_name):
        """
        Downloads the YOLOv8 model weights if not already present.
        Args:
            weight_name (str): Name of the YOLOv8 model weights (e.g., 'yolov8n.pt').
        Returns:
            str: Path to the YOLOv8 model weights file.
        """
        if os.path.exists(weight_name):
            return weight_name

        print(f"Downloading YOLOv8 model weights: {weight_name}...")
        url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{weight_name}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(weight_name, "wb") as f:
                f.write(response.content)
            print(f"Model weights downloaded: {weight_name}")
            return weight_name
        else:
            raise RuntimeError(f"Failed to download model weights from {url} (status code: {response.status_code})")

    def detect_objects(self, image_path):
        """
        Detect objects in the provided image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            tuple: Original image (NumPy array), detection results (list of dictionaries).
        """
        # Run object detection
        results = self.model(image_path)

        # Extract detections
        detections = []
        for box in results[0].boxes:
            detections.append({
                "bbox": box.xyxy[0].cpu().numpy().tolist(),  # Bounding box coordinates [x1, y1, x2, y2]
                "class_id": int(box.cls[0]),  # Class ID
                "confidence": float(box.conf[0])  # Confidence score
            })

        # Get the original image
        orig_image = results[0].orig_img

        return orig_image, detections

    def visualize_results(self, orig_image, detections):
        """
        Visualize YOLOv8 detection results.
        Args:
            orig_image (np.ndarray): Original image in BGR format.
            detections (list): Detection results.
        """
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        # Convert the image from BGR to RGB for correct display with matplotlib
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        # Plot the image with bounding boxes
        plt.imshow(image_rgb)
        ax = plt.gca()

        # Draw the bounding boxes and labels on the image
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]  # Get bounding box coordinates
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            # Safely handle class labels
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"

            # Draw the bounding box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
            ax.text(x1, y1 - 10, label, color="red", fontsize=10, backgroundcolor="white")

        plt.axis('off')  # Remove axis
        plt.show()

