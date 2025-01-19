# api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from typing import List
from pydantic import BaseModel
import logging
import os
import torch
import numpy as np
import cv2
from tempfile import TemporaryDirectory
import io
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Object Detection API")


# Response model for detection results
class DetectionResponse(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]


# Environment variables
MODEL_TYPE = os.getenv("MODEL_TYPE")
if not MODEL_TYPE:
    raise ValueError("MODEL_TYPE environment variable must be set")

CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/cache")
cache_path = Path(CACHE_DIR)

try:
    cache_path.mkdir(parents=True, exist_ok=True)
    if not os.access(CACHE_DIR, os.W_OK):
        raise PermissionError(f"Cache directory {CACHE_DIR} is not writable")
except Exception as e:
    logger.error(f"Failed to create or access cache directory: {str(e)}")
    raise RuntimeError(f"Cache directory initialization failed: {e}")

logger.info(f"Initialized with MODEL_TYPE={MODEL_TYPE}, CACHE_DIR={CACHE_DIR}")
# Model cache to avoid reloading models
_model_cache = {}


# Lazy loading function for models
def load_model(model_type: str):
    if model_type in _model_cache:
        return _model_cache[model_type]

    try:
        logger.info(f"Loading model: {model_type}")
        model_class = {
            "detr": "app.models.detr.detr.DETR",
            "detectron2": "app.models.detectron2.detectron2_model.Detectron2",
            "detrclip": "app.models.detrclip.detrclip.DETRCLIP",
            "groundingdino": "app.models.groundingdino.groundingdino.GroundingDINO",
            "kosmos2": "app.models.kosmos2.kosmos2.Kosmos2",
            "owlvit": "app.models.owlvit.owlvit.OWLVit",
            "rtdetr": "app.models.rtdetr.rtdetr.RTDETR",
            "sam2": "app.models.sam2.sam2.SAM2",
            "groundedsam2florence2": "app.models.groundedsam2florence2.groundedsam2florence2.GroundedSAM2Florence2",
            "groundedsam2": "app.models.groundedsam2.groundedsam2.GroundedSAM2",
            "yolov5": "app.models.yolov5.yolov5.YOLOv5",
            "yolov6": "app.models.yolov6.yolov6.YOLOv6",
            "yolov7": "app.models.yolov7.yolov7.YOLOv7",
            "yolov8": "app.models.yolov8.yolov8.YOLOv8",
            "yolov10": "app.models.yolov10.yolov10.YOLOv10",
            "yolov11": "app.models.yolov11.yolov11.YOLOv11",
            "yolox": "app.models.yolox.yolox.YOLOX",
        }

        if model_type not in model_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        module_path, class_name = model_class[model_type].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model = getattr(module, class_name)()

        _model_cache[model_type] = model
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {e}")


# Initialize the model
try:
    model = load_model(MODEL_TYPE)
except RuntimeError as e:
    logger.error(f"Model initialization failed: {e}")
    raise


# Endpoint: Health Check
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns the status of the server, model type, and resource usage.
    """
    return {
        "status": "healthy",
        "model_type": MODEL_TYPE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# Endpoint: Object Detection
@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for object detection.
    Accepts an image file and returns detection results.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Unsupported file type. Please upload an image."
            )

        with TemporaryDirectory() as tmp_dir:
            # Save the uploaded file temporarily
            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Perform detection
            results = model.detect_objects(str(file_path))

            # Convert numpy arrays to lists for JSON serialization
            response = {
                "boxes": results["boxes"].tolist(),
                "scores": results["scores"].tolist(),
                "labels": results["labels"].tolist(),
            }

            return DetectionResponse(**response)
    except Exception as e:
        logger.error(f"Error processing detection request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def visualize_objects(file: UploadFile = File(...)):
    """
    Endpoint to visualize object detection results.
    Accepts an image file and returns the visualized image using the model's visualization method.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        with TemporaryDirectory() as tmp_dir:
            # Save the uploaded file temporarily
            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Perform detection and visualization
            try:
                results = model.detect_objects(str(file_path))
                visualized_img = model.visualize_results(results)

                # Convert numpy array to bytes
                success, img_encoded = cv2.imencode(
                    ".png", cv2.cvtColor(visualized_img, cv2.COLOR_RGB2BGR)
                )
                if not success:
                    raise HTTPException(
                        status_code=500, detail="Failed to encode output image"
                    )

                # Create bytes stream
                img_bytes = io.BytesIO(img_encoded.tobytes())
                img_bytes.seek(0)

                return StreamingResponse(
                    img_bytes,
                    media_type="image/png",
                    headers={"Content-Disposition": f"attachment; filename=output.png"},
                )

            except Exception as e:
                logger.error(f"Detection or visualization failed: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Detection or visualization failed: {str(e)}",
                )

    except Exception as e:
        logger.error(f"Error processing visualization request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
