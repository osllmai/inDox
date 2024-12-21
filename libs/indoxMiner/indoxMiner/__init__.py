# __init__.py

from .data_extraction.extractor import Extractor
from .data_extraction.schema import ExtractorSchema, Schema
from .data_extraction.auto_schema import AutoDetectedField, AutoExtractionRules, AutoSchema
from .data_extraction.extraction_results import ExtractionResult, ExtractionResults
from .data_extraction.fields import Field, ValidationRule, FieldType
from .data_extraction.loader import DocumentProcessor, ProcessingConfig

from .data_extraction.llms import (
    OpenAi,
    Anthropic,
    NerdTokenApi,
    AsyncNerdTokenApi,
    AsyncOpenAi,
    Ollama,
    IndoxApi,
)
from .object_detection.models.Kosmos_2.kosmos2 import Kosmos2ObjectDetector
from .object_detection.models.RT_DETR.rtdetr import RTDETRModel
from .object_detection.models.DETR.detr import DETRModel
from .object_detection.models.DETR_CLIP.detr_clip_model import DETRCLIPModel
from .object_detection.models.llava_next.llava_next import LLaVANextObjectDetector
from .object_detection.models.GroundingDINO.groundingdino import GroundingDINOModel
from .object_detection.models.YOLOX.yolox_model import YOLOXModel
from .object_detection.models.OWL_ViT.owlvit import OWLVitModel
from .object_detection.models.Detectron2.detectron2_model import Detectron2Model
from .object_detection.models.SAM2.sam2_model import SAM2Model
from .object_detection.models.YOLOv5.yolov5_model import YOLOv5Model
from .object_detection.models.YOLOv6.yolov6_model import YOLOv6Model
from .object_detection.models.YOLOv7.yolov7_model import YOLOv7Model
from .object_detection.models.YOLOv8.yolov8_model import YOLOv8Model
from .object_detection.models.YOLOv10.yolov10_model import YOLOv10Model
from .object_detection.models.YOLOv11.yolov11_model import YOLOv11Model

__all__ = [
    # Extractor and schema related
    "Extractor",
    "ExtractorSchema",
    "Schema",  # For accessing predefined schemas like Passport, Invoice, etc.
    "ExtractionResult",
    "ExtractionResults",
    "Field",
    "ValidationRule",
    "FieldType",
    # Document processing related
    "DocumentProcessor",
    "ProcessingConfig",
    # llms
    "OpenAi",
    "Anthropic",
    "NerdTokenApi",
    "AsyncNerdTokenApi",
    "AsyncOpenAi",
    "Ollama",
    "IndoxApi",
]

# Package metadata
__version__ = "0.0.12"
__author__ = "IndoxMiner Team"
__description__ = "A comprehensive document extraction and processing library"
