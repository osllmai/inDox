import cv2
import torch
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import requests
import os


class Detectron2:
    def __init__(
        self,
        config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        score_thresh=0.5,
        device=None,
    ):
        """
        Initializes the Detectron2 model.
        :param config_path: Path to the model config file from the model zoo.
        :param score_thresh: Threshold for displaying detected objects.
        :param device: Device to use ('cpu' or 'cuda'). If None, automatically detects GPU.
        """
        # config_path = self._download_config(config=config)
        # url = f"https://raw.githubusercontent.com/facebookresearch/detectron2/refs/heads/main/configs/{config}"

        config_path = "COCO-Detection/faster_rcnn_R_50_C4_1x"
        self.cfg = get_cfg()

        if device is None:
            self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.cfg.MODEL.DEVICE = device

        # self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

        self.predictor = DefaultPredictor(self.cfg)

    # def _download_config(self, config):
    #     url = f"https://raw.githubusercontent.com/facebookresearch/detectron2/refs/heads/main/configs/{config}"

    #     filename = os.path.basename(config)  # This is safer than url.split("/")[-1]

    #     config_response = requests.get(url)
    #     config_response.raise_for_status()  # Check if download was successful

    #     config_path = os.path.join(os.getcwd(), filename)

    #     # Write the content
    #     with open(config_path, "wb") as f:
    #         f.write(config_response.content)

    #     return config_path

    def detect_objects(self, image_path):
        """
        Detect objects in the given image.
        :param image_path: Path to the input image.
        :return: A dictionary containing instances with predicted bounding boxes, classes, and scores.
        """
        img = cv2.imread(image_path)
        outputs = self.predictor(img)

        # Store original image in outputs for visualization
        outputs["orig_img"] = img
        return outputs

    def visualize_results(self, outputs):
        """
        Visualize the results of object detection on the image using matplotlib.
        :param outputs: The output dictionary from detect_objects() method.
        """
        img = outputs["orig_img"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()

        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.tensor[i].numpy()
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                )
                ax.add_patch(rect)

                # Add labels above bounding boxes if available
                if classes is not None and scores is not None:
                    class_id = classes[i].item()
                    score_val = scores[i].item()
                    # ax.text(x1, y1, f"Class: {class_id}, Score: {score_val:.2f}",
                    #         fontsize=10, color='white', backgroundcolor='red')

        plt.axis("off")
        plt.show()
