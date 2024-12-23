import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

class YOLOv5Model:
    def __init__(self, model_name='yolov5s', device='cuda'):
        """
        Initialize the YOLOv5 model.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Load YOLOv5 model from Torch Hub with explicit trust
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, trust_repo=True).to(self.device)
        self.model.eval()

    def detect_objects(self, image_path, conf_thres=0.25, iou_thres=0.45):
        """
        Perform object detection on an input image.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Perform inference
        self.model.conf = conf_thres
        self.model.iou = iou_thres
        results = self.model(img_bgr[..., ::-1])  # Convert BGR to RGB for inference

        return results, img_bgr


    def visualize_results(self, results, img_bgr):
        """
        Visualize the detection results on the image using matplotlib.

        Args:
            results: Inference results returned by `self.detect_objects`.
            img_bgr: Original BGR image on which detection was performed.
        """
        print(f"Visualizing results for {len(results.xyxy[0])} detections...")
        print(img_bgr.shape)
        # YOLOv5 results are in results.xyxy[0] as [x1, y1, x2, y2, conf, class]
        detections = results.xyxy[0].cpu().numpy() # Debugging output
        print(f"Detections: {detections}")  # Debugging output

        # Convert image to RGB for plotting
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()

        # Plot each detection
        for x1, y1, x2, y2, conf, cls in detections:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            class_name = self.model.names[int(cls)]
            ax.text(x1, y1 - 5, f"{class_name} {conf:.2f}",
                    color='white', fontsize=10, backgroundcolor='red')

        plt.axis('off')
        plt.title("YOLOv5 Detection Results")
        plt.show()
        plt.close()  # Explicitly close the figure

