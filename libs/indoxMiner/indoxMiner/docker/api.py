# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from models.DETR.detr import DETR
from models.Detectron2.detectron2_model import Detectron2
from models.detr_clip.detr_clip import DETRCLIP
from models.GroundingDINO.groundingdino import GroundingDINO
from models.kosmos2.kosmos2 import Kosmos2
from models.owlvit.owlvit import OWLVit
from models.rtdetr.rtdetr import RTDETR
from models.sam2.sam2 import SAM2
from models.YOLO.YOLOv5.yolov5 import YOLOv5
from models.YOLO.YOLOv6.yolov6 import YOLOv6
from models.YOLO.YOLOv7.yolov7 import YOLOv7
from models.YOLO.YOLOv8.yolov8 import YOLOv8
from models.YOLO.YOLOv10.yolov10 import YOLOv10
from models.YOLO.YOLOv11.yolov11 import YOLOv11
from models.YOLO.YOLOX.yolox import YOLOX
import torch

app = FastAPI()

# Initialize model based on environment variable
MODEL_TYPE = os.getenv("MODEL_TYPE", "detr")
model_map = {
    "detr": DETR,
    "yolov8": YOLOv8,
    "detectron2": Detectron2,
    "detrclip": DETRCLIP,
    "groundingdino": GroundingDINO,
    "kosmos2": Kosmos2,
    "owlvit": OWLVit,
    "rtdetr": RTDETR,
    "sam2": SAM2,
    "yolov5": YOLOv5,
    "yolov6": YOLOv6,
    "yolov7": YOLOv7,
    "yolov10": YOLOv10,
    "yolov11": YOLOv11,
    "yolox": YOLOX,
}

if MODEL_TYPE not in model_map:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

model = model_map[MODEL_TYPE](device="cuda" if torch.cuda.is_available() else "cpu")


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Perform detection
        results = model.detect_objects(file_path)

        # Clean up
        os.remove(file_path)

        return JSONResponse(
            content={
                "boxes": results["boxes"].tolist(),
                "scores": results["scores"].tolist(),
                "labels": results["labels"].tolist(),
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
