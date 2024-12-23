import os
import cv2
import torch
import numpy as np
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils.visualize import vis
from yolox.data.data_augment import preproc
import matplotlib.pyplot as plt


class YOLOXModel:
    def __init__(self, exp_file, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.exp = get_exp(exp_file, None)
        self.exp.num_classes = len(COCO_CLASSES)  # Ensure num_classes matches the dataset
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = self.exp.get_model().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        model.eval()
        return model

    def preprocess(self, image_path):
        image = cv2.imread(image_path)  # BGR format
        input_size = self.exp.test_size
        tensor_image, scale = preproc(image, input_size)
        tensor_image = torch.from_numpy(tensor_image).unsqueeze(0).float().to(self.device)
        return image, tensor_image, scale

    def detect_objects(self, image_path, conf_thre=0.25, nms_thre=0.45):
        # Preprocess the image
        original_image, tensor_image, scale = self.preprocess(image_path)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(tensor_image)
            outputs = postprocess(
                outputs,
                num_classes=len(COCO_CLASSES),
                conf_thre=conf_thre,
                nms_thre=nms_thre,
            )
        return original_image, outputs, scale

    def visualize_results(self, image, outputs, scale, save_path="result.jpg"):
        if outputs is None or len(outputs[0]) == 0:
            print("No detections found.")
            return None

        bboxes = outputs[0][:, 0:4] / scale
        scores = outputs[0][:, 4] * outputs[0][:, 5]
        class_ids = outputs[0][:, 6].to(dtype=torch.int).cpu().numpy()  # Convert to integer and NumPy array
        result_image = vis(image, bboxes.cpu().numpy(), scores.cpu().numpy(), class_ids, conf=0.3, class_names=COCO_CLASSES)
        cv2.imwrite(save_path, result_image)

        # Display the image
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Detections")
        plt.axis("off")
        plt.show()

        return result_image

