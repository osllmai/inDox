import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
import os

class SAM2Model:
    def __init__(self, checkpoint_path, config_path="sam2_hiera_l.yaml"):
        # Set device (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SAM2 model
        self.model = build_sam2(config_path, checkpoint_path, device=self.device, apply_postprocessing=False)
        
        # Set up the mask generator for automatic mask generation
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def detect_objects(self, image_path):
        # Read and preprocess the image
        image_bgr = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate segmentation masks using SAM2
        sam2_result = self.mask_generator.generate(image_rgb)

        # Create detections from SAM2 result
        detections = sv.Detections.from_sam(sam_result=sam2_result)
        return image_bgr, detections

    def visualize_results(self, image_bgr, detections):
        # Convert the image from BGR to RGB before displaying
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Create a mask annotator for visualizing the segmented masks
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        
        # Annotate the image with bounding boxes
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image_with_boxes = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        
        # Optionally, annotate masks as well
        # annotated_image = mask_annotator.annotate(scene=annotated_image_with_boxes, detections=detections)
        
        # Visualize the results using Matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_with_boxes)
        plt.axis("off")  # Remove axis
        plt.show()

    def visualize_masks(self, sam2_result):
        # Extract masks from SAM2 result and display them
        masks = [mask['segmentation'] for mask in sorted(sam2_result, key=lambda x: x['area'], reverse=True)]
        
        # Plot the individual masks
        sv.plot_images_grid(
            images=masks[:16],
            grid_size=(4, 4),
            size=(12, 12)
        )