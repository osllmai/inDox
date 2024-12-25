# import subprocess
# import sys
# import torch
# from typing import Literal, Optional, Union


# class ObjectDetection:
#     """
#     A factory class for creating object detection model instances.
#     """

#     def __init__(self):
#         """Initialize the ObjectDetection class."""
#         self.available_models = [
#             "detectron2",
#             "detr",
#             "detrclip",
#             "groundingdino",
#             "kosmos2",
#             "owlvit",
#             "rtdetr",
#             "sam2",
#         ]

#     def _install_detectron2(self) -> bool:
#         """
#         Install Detectron2 if not already installed.
#         Returns:
#             bool: True if installation successful or already installed, False otherwise
#         """
#         try:
#             import detectron2
#             return True
#         except ImportError:
#             print("Detectron2 not found. Installing...")
#             try:
#                 # Install torch and torchvision first if not installed
#                 subprocess.check_call(
#                     [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
#                 )

#                 # Install detectron2
#                 subprocess.check_call(
#                     [
#                         sys.executable,
#                         "-m",
#                         "pip",
#                         "install",
#                         "git+https://github.com/facebookresearch/detectron2.git",
#                     ]
#                 )
#                 return True
#             except subprocess.CalledProcessError as e:
#                 print(f"Error installing Detectron2: {e}")
#                 return False

#     def load_model(
#             self,
#             model: Literal[
#                 "detectron2",
#                 "detr",
#                 "detrclip",
#                 "groundingdino",
#                 "kosmos2",
#                 "owlvit",
#                 "rtdetr",
#                 "sam2",
#             ],
#             device: Optional[str] = None,
#             **kwargs,
#         ) -> Union[
#             "DETRModel",
#             "Detectron2Model",
#             "DETRCLIPModel",
#             "GroundingDINOModel",
#             "Kosmos2Model",
#             "OWLVitModel",
#             "RTDETRModel",
#             "SAM2Model",
#         ]:
#         """
#         Load and return the specified object detection model.

#         Args:
#             model (str): Model to load ("detectron2", "detr", "detrclip", "groundingdino", "kosmos2", "owlvit", or "rtdetr")
#             device (str, optional): Device to use ('cuda' or 'cpu')
#             **kwargs: Additional arguments to pass to the model constructor

#         Returns:
#             The loaded model instance
#         """
#         if model not in self.available_models:
#             raise ValueError(
#                 f"Unsupported model: {model}. Available models: {self.available_models}"
#             )

#         # Set default device if not specified
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         if model == "detectron2":
#             if not self._install_detectron2():
#                 raise RuntimeError("Failed to install Detectron2")

#             from .models.detectron2.detectron2_model import Detectron2Model
#             return Detectron2Model(device=device, **kwargs)

#         elif model == "detr":
#             from .models.detr.detr import DETRModel
#             return DETRModel(device=device, **kwargs)

#         elif model == "detrclip":
#             from .models.detrclip.detrclip import DETRCLIPModel
#             return DETRCLIPModel(device=device, **kwargs)

#         elif model == "groundingdino":
#             from .models.groundingdino.groundingdino import GroundingDINOModel
#             config_path = kwargs.get("config_name_or_path", "GroundingDINO_SwinT_OGC.py")
#             weights_path = kwargs.get("weights_name_or_path", "groundingdino_swint_ogc.pth")

#             return GroundingDINOModel(
#                 config_name_or_path=config_path,
#                 weights_name_or_path=weights_path,
#                 device=device,
#                 box_threshold=kwargs.get("box_threshold", 0.35),
#                 text_threshold=kwargs.get("text_threshold", 0.25),
#             )

#         elif model == "kosmos2":
#             from .models.kosmos2.kosmos2 import Kosmos2Model
#             model_id = kwargs.get("model_id", "microsoft/kosmos-2-patch14-224")
#             return Kosmos2Model(model_id=model_id, device=device)

#         elif model == "owlvit":
#             from .models.owlvit.owlvit import OWLVitModel
#             return OWLVitModel()

#         elif model == "rtdetr":
#             from .models.rtdetr.rtdetr import RTDETRModel
#             checkpoint = kwargs.get("checkpoint", "PekingU/rtdetr_r50vd_coco_o365")
#             return RTDETRModel(checkpoint=checkpoint, device=device)
#         elif model == "sam2":

#             # Load SAM2 model
#             from .models.sam2.sam2 import SAM2Model
#             config_path = kwargs.get("config_name_or_path", None)
#             weights_path = kwargs.get("weights_name_or_path", None)
#             return SAM2Model(config_name_or_path=config_path, weights_name_or_path=weights_path, device=device)
