import subprocess
import sys
import os
import cv2
import torch
import matplotlib.pyplot as plt

class Detectron2:
    def __init__(self, config_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", score_thresh=0.5, device=None):
        """
        Initializes the Detectron2 model and prepares it for inference.
        
        :param config_name: Name of the config file from the model zoo.
        :param score_thresh: Threshold for displaying detected objects.
        :param device: Device to use ('cpu' or 'cuda'). If None, automatically detects GPU.
        """
        self.config_name = config_name
        self.score_thresh = score_thresh
        self.device = device

        # Install detectron2 if not already installed
        self._install_detectron2()

        # Now import the necessary modules after installation
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        global Visualizer, ColorMode
        from detectron2.utils.visualizer import Visualizer, ColorMode

        # Download config files and prepare the model
        self.cfg = self._configure_model(model_zoo, get_cfg, DefaultPredictor)

    def _install_detectron2(self):
        """Installs Detectron2 if it's not already installed."""
        try:
            import detectron2
        except ImportError:
            print("Detectron2 library not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git'])
            print("Detectron2 installed successfully.")

    def _configure_model(self, model_zoo, get_cfg, DefaultPredictor):
        """Configures the Detectron2 model with the chosen config file."""
        # Load configuration
        cfg = get_cfg()

        # Set device to 'cuda' if available, otherwise 'cpu'
        if self.device is None:
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            cfg.MODEL.DEVICE = self.device

        # Merge configuration from the model zoo
        cfg.merge_from_file(model_zoo.get_config_file(self.config_name))

        # Set model weights and score threshold for inference
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_name)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh

        # Create the predictor
        self.predictor = DefaultPredictor(cfg)
        return cfg

    def detect_objects(self, image_path):
        """
        Detect objects in the given image.
        
        :param image_path: Path to the input image.
        :return: A dictionary containing instances with predicted bounding boxes, classes, and scores.
        """
        img = cv2.imread(image_path)

        # Run inference
        outputs = self.predictor(img)
        
        # Store the original image in the output dictionary for visualization
        outputs["orig_img"] = img
        return outputs

    def visualize_results(self, outputs):
        """
        Visualizes the results of object detection on the image using matplotlib.
        
        :param outputs: The output dictionary from detect_objects() method.
        """
        img_rgb = cv2.cvtColor(outputs["orig_img"], cv2.COLOR_BGR2RGB)

        # Prepare the Visualizer
        v = Visualizer(img_rgb,
                       metadata=None,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE)

        # Draw predictions on the image
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Display the results
        plt.figure(figsize=(12, 8))
        plt.imshow(out.get_image())
        plt.axis('off')
        plt.title("Object Detection Results")
        plt.show()

        return out.get_image()

    def save_results(self, image, output_path="output.jpg"):
        """Saves the output image with drawn bounding boxes."""
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Results saved to {output_path}")