from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path
import torch
import numpy as np
import psutil
from pydantic import BaseModel
from tempfile import TemporaryDirectory
from typing import List
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Object Detection API")


# Define response model
class DetectionResponse(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]


# Environment variables for model configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "detr")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/cache")

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


# Lazy loading function for models
def load_model(model_type: str):
    if model_type == "detr":
        from models.detr.detr import DETR

        return DETR
    elif model_type == "detectron2":
        from models.detectron2.detectron2_model import Detectron2

        return Detectron2
    elif model_type == "detrclip":
        from models.detrclip.detrclip import DETRCLIP

        return DETRCLIP
    elif model_type == "groundingdino":
        from models.groundingdino.groundingdino import GroundingDINO

        return GroundingDINO
    elif model_type == "kosmos2":
        from models.kosmos2.kosmos2 import Kosmos2

        return Kosmos2
    elif model_type == "owlvit":
        from models.owlvit.owlvit import OWLVit

        return OWLVit
    elif model_type == "rtdetr":
        from models.rtdetr.rtdetr import RTDETR

        return RTDETR
    elif model_type == "sam2":
        from models.sam2.sam2 import SAM2

        return SAM2
    elif model_type == "yolov5":
        from models.yolov5.yolov5 import YOLOv5

        return YOLOv5
    elif model_type == "yolov6":
        from models.yolov6.yolov6 import YOLOv6

        return YOLOv6
    elif model_type == "yolov7":
        from models.yolov7.yolov7 import YOLOv7

        return YOLOv7
    elif model_type == "yolov8":
        from models.yolov8.yolov8 import YOLOv8

        return YOLOv8
    elif model_type == "yolov10":
        from models.yolov10.yolov10 import YOLOv10

        return YOLOv10
    elif model_type == "yolov11":
        from models.yolov11.yolov11 import YOLOv11

        return YOLOv11
    elif model_type == "yolox":
        from models.yolox.yolox import YOLOX

        return YOLOX
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Initialize the model
try:
    model_class = load_model(MODEL_TYPE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model_class(device=device, cache_dir=CACHE_DIR)
    model = model_class(device=device)

    logger.info(f"Initialized {MODEL_TYPE} model on {device}")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise RuntimeError(f"Failed to initialize model: {e}")


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for object detection.
    Accepts an image file and returns detection results.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        with TemporaryDirectory() as tmp_dir:
            # Save the uploaded file temporarily
            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

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
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns the status of the server, model type, and resource usage.
    """
    return {
        "status": "healthy",
        "model_type": MODEL_TYPE,
        "device": str(model.device),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "gpu_memory_mb": (
            torch.cuda.memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else None
        ),
    }


@app.post("/visualize")
async def visualize_objects(file: UploadFile = File(...)):
    """
    Endpoint to visualize object detection results.
    Accepts an image file and returns the image with bounding boxes drawn.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        with TemporaryDirectory() as tmp_dir:
            # Save the uploaded file temporarily
            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Perform detection
            results = model.detect_objects(str(file_path))

            # Create a temporary file for the visualization
            with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                # Call visualize_results
                plt.figure(figsize=(12, 8))
                model.visualize_results(results)
                plt.savefig(temp_file.name)
                plt.close()  # Close the figure to free memory

                # Return the visualized image
                return FileResponse(temp_file.name, media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing visualization request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
