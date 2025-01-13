import os
import cv2
import torch
import numpy as np
import subprocess
import sys
import requests
import glob
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class YOLOX:
    def __init__(self, model_name="yolox_s", device="cuda"):
        """
        Initializes the YOLOX model.

        Args:
            model_name (str): Name of the YOLOX model (e.g., 'yolox_s', 'yolox_m').
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # # Ensure YOLOX is installed
        # self._install_yolox()

        # Download or locate the experiment and weights files
        self.exp_file = self._get_file(self.model_name, "exp")
        self.model_file = self._get_file(self.model_name, "weights")

        # Load YOLOX experiment and model
        self.exp = self._load_experiment(self.exp_file)
        self.exp.num_classes = self._get_coco_classes_length()
        self.model = self._load_model(self.model_file)

    # def _install_yolox(self):
    #     """
    #     Ensures that YOLOX library is installed.
    #     """
    #     try:
    #         subprocess.run(
    #             ["yolo", "--help"],
    #             check=True,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #         )
    #     except FileNotFoundError:
    #         print("YOLOX library not found. Installing...")
    #         try:
    #             subprocess.check_call(
    #                 [
    #                     sys.executable,
    #                     "-m",
    #                     "pip",
    #                     "install",
    #                     "git+https://github.com/Megvii-BaseDetection/YOLOX.git",
    #                 ]
    #             )
    #         except subprocess.CalledProcessError as e:
    #             raise RuntimeError(f"Failed to install YOLOX library: {e}")

    def _get_file(self, model_name, file_type):
        """
        Downloads the experiment or weights file for the specified YOLOX model.

        Args:
            model_name (str): Name of the YOLOX model (e.g., 'yolox_s').
            file_type (str): Type of file ('exp' or 'weights').

        Returns:
            str: Path to the file.
        """
        if file_type == "exp":
            url = f"https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/exps/default/{model_name}.py"
            file_name = f"{model_name}.py"
        elif file_type == "weights":
            url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_name}.pth"
            file_name = f"{model_name}.pth"
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        # Check if the file already exists
        if os.path.exists(file_name):
            return file_name

        # Download the file
        print(f"Downloading {file_type} file for {model_name}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(response.content)
            print(f"{file_type.capitalize()} file downloaded: {file_name}")
            return file_name
        else:
            raise RuntimeError(
                f"Failed to download {file_type} file from {url} (status code: {response.status_code})"
            )

    def _load_experiment(self, exp_file):
        """
        Loads the YOLOX experiment configuration.

        Args:
            exp_file (str): Path to the experiment file.

        Returns:
            YOLOX experiment object.
        """
        from yolox.exp import get_exp

        return get_exp(exp_file, None)

    def _get_coco_classes_length(self):
        """
        Returns the length of COCO_CLASSES.

        Returns:
            int: Number of COCO classes.
        """
        from yolox.data.datasets import COCO_CLASSES

        return len(COCO_CLASSES)

    def _load_model(self, model_path):
        """
        Loads the YOLOX model from the specified path.

        Args:
            model_path (str): Path to the model weights file.

        Returns:
            torch.nn.Module: Loaded YOLOX model.
        """
        from yolox.utils import fuse_model

        model = self.exp.get_model().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        model.eval()
        return model

    def preprocess(self, image_path):
        """
        Preprocesses the input image for YOLOX inference.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Original image, preprocessed tensor image, and scale factor.
        """
        from yolox.data.data_augment import preproc

        image = cv2.imread(image_path)  # BGR format
        input_size = self.exp.test_size
        tensor_image, scale = preproc(image, input_size)
        tensor_image = (
            torch.from_numpy(tensor_image).unsqueeze(0).float().to(self.device)
        )
        return image, tensor_image, scale

    def detect_objects(self, image_path, conf_thre=0.25, nms_thre=0.45):
        """
        Detects objects in the input image.

        Args:
            image_path (str): Path to the input image.
            conf_thre (float): Confidence threshold for detections.
            nms_thre (float): Non-maximum suppression threshold.

        Returns:
            dict: A dictionary containing the original image, detection outputs, and metadata.
        """
        from yolox.utils import postprocess

        # Preprocess the image
        original_image, tensor_image, scale = self.preprocess(image_path)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(tensor_image)
            outputs = postprocess(
                outputs,
                num_classes=self.exp.num_classes,
                conf_thre=conf_thre,
                nms_thre=nms_thre,
            )

        # If no outputs, return empty results
        if outputs is None or len(outputs[0]) == 0:
            print("No detections found.")
            packed_results = {
                "original_image": original_image,
                "detections": [],
                "metadata": {
                    "conf_threshold": conf_thre,
                    "nms_threshold": nms_thre,
                    "num_detections": 0,
                },
            }
            return packed_results

        # Adjust the bounding boxes to the original image size
        outputs[0][:, 0:4] /= scale

        # Pack results into a dictionary
        detections = [
            {
                "bbox": box[:4].cpu().numpy().tolist(),
                "confidence": float(box[4].cpu()),
                "class_id": int(box[6].cpu()),
            }
            for box in outputs[0]
        ]

        packed_results = {
            "original_image": original_image,
            "detections": detections,
            "metadata": {
                "conf_threshold": conf_thre,
                "nms_threshold": nms_thre,
                "num_detections": len(detections),
            },
        }

        return packed_results


    def visualize_results(self, packed_results, save_path="result.jpg"):
        """
        Visualizes detection results on the image.

        Args:
            packed_results (dict): Packed results containing the original image and detections.
            save_path (str): Path to save the annotated image.

        Returns:
            np.ndarray: Annotated image with detections.
        """
        from yolox.utils.visualize import vis
        from yolox.data.datasets import COCO_CLASSES

        original_image = packed_results["original_image"]
        detections = packed_results["detections"]

        if not detections:
            print("No detections to visualize.")
            return None


        bboxes = np.array([det["bbox"] for det in detections])
        scores = np.array([det["confidence"] for det in detections])
        class_ids = np.array([det["class_id"] for det in detections])

        result_image = vis(original_image, bboxes, scores, class_ids, conf=0.3, class_names=COCO_CLASSES)
        cv2.imwrite(save_path, result_image)

        # Display the image
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Detections")
        plt.axis("off")
        plt.show()

