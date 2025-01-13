import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys


class YOLOv5:
    def __init__(self, model_name="yolov5s", device="cuda"):
        """
        Initialize the YOLOv5 model.
        Args:
            model_name (str): Name of the YOLOv5 model (e.g., 'yolov5s', 'yolov5m', 'yolov5l').
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load YOLOv5 model from Torch Hub with explicit trust
        self._install_yolov5()
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5", model_name, pretrained=True, trust_repo=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading YOLOv5 model '{model_name}': {e}")

    def _install_yolov5(self):
        """
        Install YOLOv6 from the official repository if not already installed.
        """
        if not os.path.exists("YOLOv6"):
            print("Cloning YOLOv6 repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/meituan/YOLOv6"], check=True
            )
            print("YOLOv6 repository cloned.")
        os.chdir("YOLOv6")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        os.chdir("..")

    def detect_objects(self, image_path, conf_thres=0.25, iou_thres=0.45):
        """
        Perform object detection on an input image.
        Args:
            image_path (str): Path to the input image.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): Intersection-over-union threshold for NMS.
        Returns:
            tuple: Original image (BGR format), detections as a list of dictionaries.
        """
        # Read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Set model thresholds
        self.model.conf = conf_thres
        self.model.iou = iou_thres

        # Perform inference
        results = self.model(img_bgr[..., ::-1])  # Convert BGR to RGB

        # Parse results
        detections = []
        # Access the first result's boxes and other attributes
        if len(results) > 0:
            boxes = results[0].boxes  # Get boxes of first image
            for box in boxes:
                # Get box coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # box coordinates
                conf = box.conf[0].item()  # confidence score
                cls = box.cls[0].item()  # class id

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": self.model.names[int(cls)],
                    }
                )

        return img_bgr, detections

    def visualize_results(self, img_bgr, detections):
        """
        Visualize the detection results on the image using matplotlib.
        Args:
            img_bgr (np.ndarray): Original BGR image.
            detections (list): List of detection dictionaries.
        """
        if len(detections) == 0:
            print("No detections to visualize.")
            return

        # Convert image to RGB for plotting
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Create the figure and axis once
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)

        # Plot each detection
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class_name"]

            # Draw bounding box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x1,
                y1 - 5,
                f"{class_name} {conf:.2f}",
                color="white",
                fontsize=10,
                backgroundcolor="red",
            )

        # Customize plot and display
        ax.axis("off")
        plt.title("YOLOv5 Detection Results")
        plt.show()


# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# import os
# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# class YOLOv5:
#     def __init__(self, model_name="yolov5s", device="cuda"):
#         """
#         Initialize the YOLOv5 model.
#         Args:
#             model_name (str): Name of the YOLOv5 model (e.g., 'yolov5s', 'yolov5m', 'yolov5l').
#             device (str): Device to run the model on ('cuda' or 'cpu').
#         """
#         from ultralytics import YOLO

#         self.device = device if torch.cuda.is_available() else "cpu"

#         # Load YOLO model from ultralytics package
#         try:
#             self.model = YOLO(model_name).to(self.device)
#             self.model.eval()
#         except Exception as e:
#             raise RuntimeError(f"Error loading YOLOv5 model '{model_name}': {e}")

#     def detect_objects(self, image_path, conf_thres=0.25, iou_thres=0.45):
#         """
#         Perform object detection on an input image.
#         Args:
#             image_path (str): Path to the input image.
#             conf_thres (float): Confidence threshold for detections.
#             iou_thres (float): Intersection-over-union threshold for NMS.
#         Returns:
#             tuple: Original image (BGR format), detections as a list of dictionaries.
#         """
#         # Read image
#         img_bgr = cv2.imread(image_path)
#         if img_bgr is None:
#             raise FileNotFoundError(f"Image not found: {image_path}")

#         # Perform inference
#         results = self.model.predict(
#             img_bgr[..., ::-1], conf=conf_thres, iou=iou_thres, device=self.device
#         )

#         # Parse results
#         detections = []
#         for result in results:
#             for box in result.boxes.xyxy:
#                 x1, y1, x2, y2 = map(int, box[:4].tolist())
#                 conf = float(box[4].item())
#                 cls = int(box[5].item())
#                 detections.append(
#                     {
#                         "bbox": [x1, y1, x2, y2],
#                         "confidence": conf,
#                         "class_id": cls,
#                         "class_name": self.model.names[cls],
#                     }
#                 )

#         return img_bgr, detections

#     def visualize_results(self, img_bgr, detections):
#         """
#         Visualize the detection results on the image using matplotlib.
#         Args:
#             img_bgr (np.ndarray): Original BGR image.
#             detections (list): List of detection dictionaries.
#         """
#         if len(detections) == 0:
#             print("No detections to visualize.")
#             return

#         # Convert image to RGB for plotting
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         # Create the figure and axis once
#         fig, ax = plt.subplots(figsize=(12, 8))
#         ax.imshow(img_rgb)

#         # Plot each detection
#         for detection in detections:
#             x1, y1, x2, y2 = detection["bbox"]
#             conf = detection["confidence"]
#             class_name = detection["class_name"]

#             # Draw bounding box
#             rect = plt.Rectangle(
#                 (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
#             )
#             ax.add_patch(rect)

#             # Add label
#             ax.text(
#                 x1,
#                 y1 - 5,
#                 f"{class_name} {conf:.2f}",
#                 color="white",
#                 fontsize=10,
#                 backgroundcolor="red",
#             )

#         # Customize plot and display
#         ax.axis("off")
#         plt.title("YOLOv5 Detection Results")
#         plt.show()
