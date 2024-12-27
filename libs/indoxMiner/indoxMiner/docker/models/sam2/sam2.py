import os
import cv2
import subprocess
import sys
import requests
import matplotlib.pyplot as plt
import supervision as sv


class SAM2:
    def __init__(
        self, config_name_or_path=None, weights_name_or_path=None, device="cuda"
    ):
        """
        Initializes the SAM2 object segmentation model.

        Args:
            config_name_or_path (str, optional): Path to the config file. If None, it will be downloaded.
            weights_name_or_path (str, optional): Path to the model weights file. If None, it will be downloaded.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device

        # Ensure SAM2 is installed
        self._install_sam2()

        # Automatically download config and weight files if necessary
        self.config_path = (
            self._get_file(config_name_or_path, "config")
            if config_name_or_path
            else self._get_file("sam2_hiera_l.yaml", "config")
        )
        self.weights_path = (
            self._get_file(weights_name_or_path, "weights")
            if weights_name_or_path
            else self._get_file("sam2_hiera_large.pt", "weights")
        )

        # Load the SAM2 model (after ensuring the package is installed)
        from sam2.build_sam import build_sam2

        self.model = build_sam2(
            self.config_path,
            self.weights_path,
            device=self.device,
            apply_postprocessing=False,
        )

        # Initialize mask generator
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def _install_sam2(self):
        """
        Installs the SAM2 library if not already installed.
        """
        try:
            import sam2
        except ImportError:
            print("SAM2 not found. Installing...")
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "git+https://github.com/facebookresearch/sam2.git",
                    ]
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error installing SAM2: {e}")

    def _get_file(self, name_or_path, file_type):
        """
        Downloads the config or weights file if only a name is provided.

        Args:
            name_or_path (str): Name or path to the file.
            file_type (str): Type of file ('config' or 'weights').

        Returns:
            str: Path to the file.
        """
        if os.path.exists(name_or_path):
            return name_or_path

        # Define URLs for known files
        urls = {
            "config": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml",
            "weights": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        }

        if file_type not in urls:
            raise ValueError(f"Unknown file type: {file_type}")

        file_url = urls[file_type]
        file_name = os.path.basename(file_url)

        # Download the file
        print(f"Downloading {file_type} file: {file_name}...")
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"{file_type.capitalize()} file downloaded: {file_name}")
            return file_name
        else:
            raise RuntimeError(
                f"Failed to download {file_type} file from {file_url} (status code: {response.status_code})"
            )

    def detect_objects(self, image_path):
        """
        Detect objects in the image using SAM2.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Input image (BGR format), detections as supervision Detections object.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam2_result = self.mask_generator.generate(image_rgb)
        detections = sv.Detections.from_sam(sam_result=sam2_result)
        return image_bgr, detections

    def visualize_results(self, image_bgr, detections):
        """
        Visualize detected objects on the image.

        Args:
            image_bgr (np.ndarray): Input image in BGR format.
            detections (sv.Detections): Detections object from supervision.

        Displays the annotated image.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image_with_boxes = box_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_with_boxes)
        plt.axis("off")
        plt.show()

    def visualize_masks(self, sam2_result):
        """
        Visualize masks generated by SAM2.

        Args:
            sam2_result (list): List of masks generated by SAM2.

        Displays the individual masks in a grid.
        """
        masks = [
            mask["segmentation"]
            for mask in sorted(sam2_result, key=lambda x: x["area"], reverse=True)
        ]

        sv.plot_images_grid(images=masks[:16], grid_size=(4, 4), size=(12, 12))
