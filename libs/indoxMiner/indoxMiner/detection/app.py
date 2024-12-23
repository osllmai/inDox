from .models.Kosmos_2.kosmos2 import Kosmos2ObjectDetector
from .models.RT_DETR.rtdetr import RTDETRModel
from .models.DETR.detr import DETRModel
from .models.DETR_CLIP.detr_clip_model import DETRCLIPModel
from .models.llava_next.llava_next import LLaVANextObjectDetector
from .models.GroundingDINO.groundingdino import GroundingDINOObjectDetector
from .models.YOLOX.yolox_model import YOLOXModel
from .models.OWL_ViT.owlvit import OWLVitModel
from .models.Detectron2.detectron2_model import Detectron2Model
from .models.SAM2.sam2_model import SAM2Model
from .models.YOLOv5.yolov5_model import YOLOv5Model
from .models.YOLOv6.yolov6_model import YOLOv6Model
from .models.YOLOv7.yolov7_model import YOLOv7Model
from .models.YOLOv8.yolov8_model import YOLOv8Model
from .models.YOLOv10.yolov10_model import YOLOv10Model
from .models.YOLOv11.yolov11_model import YOLOv11Model


class IndoxObjectDetection:
    def __init__(self, model_name, device="cuda", **kwargs):
        """
        Initialize the IndoxObjectDetection class.

        Args:
            model_name (str): The name of the model to use.
            device (str): The device to use for inference, e.g., "cuda" or "cpu".
            **kwargs: Additional parameters for model initialization.
        """
        self.model_name = model_name.lower()
        self.device = device

        # Initialize the selected model
        if self.model_name == "kosmos2":
            self.model = Kosmos2ObjectDetector(device=device)
        elif self.model_name == "rtdetr":
            self.model = RTDETRModel(device=device)
        elif self.model_name == "llava":
            self.model = LLaVANextObjectDetector(device=device)
        elif self.model_name == "groundingdino":
            self._initialize_groundingdino(kwargs)
        elif self.model_name == "yolox":
            self._initialize_yolox(kwargs)
        elif self.model_name == "owlvit":
            self.model = OWLVitModel()
        elif self.model_name == "yolov8":
            self.model = YOLOv8Model(model_path=kwargs.get("model_path", "yolov8n.pt"))
        elif self.model_name == "yolov5":
            self.model = YOLOv5Model(
                model_name=kwargs.get("model_name", "yolov5s"), device=device
            )
        elif self.model_name == "yolov6":
            self.model = YOLOv6Model(device=device)
        elif self.model_name == "yolov7":
            self.model = YOLOv7Model(device=device)
        elif self.model_name == "yolov11":
            self.model = YOLOv11Model(
                model_path=kwargs.get("model_path", "yolo11n.pt"), device=device
            )
        elif self.model_name == "yolov10":
            self.model = YOLOv10Model(
                model_path=kwargs.get("model_path", "yolo10n.pt"), device=device
            )
        elif self.model_name == "detectron2":
            self._initialize_detectron2(kwargs)
        elif self.model_name == "detr":
            self.model = DETRModel(device=device)
        elif self.model_name == "detrclip":
            self.model = DETRCLIPModel(device=device)
        elif self.model_name == "sam2":
            self._initialize_sam2(kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _initialize_groundingdino(self, kwargs):
        config_path = kwargs.get("config_path")
        model_weights_path = kwargs.get("model_weights_path")
        if not config_path or not model_weights_path:
            raise ValueError(
                "Config and weights paths must be provided for Grounding DINO."
            )
        self.model = GroundingDINOObjectDetector(
            config_path=config_path,
            model_weights_path=model_weights_path,
            device=self.device,
            box_threshold=kwargs.get("box_threshold", 0.35),
            text_threshold=kwargs.get("text_threshold", 0.25),
        )

    def _initialize_yolox(self, kwargs):
        exp_file = kwargs.get("exp_file")
        model_path = kwargs.get("model_path")
        if not exp_file or not model_path:
            raise ValueError(
                "Experiment file and model path must be provided for YOLOX."
            )
        self.model = YOLOXModel(
            exp_file=exp_file, model_path=model_path, device=self.device
        )

    def _initialize_detectron2(self, kwargs):
        self.model = Detectron2Model(
            config_path=kwargs.get(
                "config_path", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            ),
            score_thresh=kwargs.get("score_thresh", 0.5),
            device=self.device,
        )

    def _initialize_sam2(self, kwargs):
        checkpoint_path = kwargs.get(
            "checkpoint_path",
            "/content/segment-anything-2/checkpoints/sam2_hiera_large.pt",
        )
        config_path = kwargs.get("config_path", "sam2_hiera_l.yaml")
        self.model = SAM2Model(checkpoint_path=checkpoint_path, config_path=config_path)

    def run(self, image_path, **kwargs):
        """
        Run the selected model on the input image.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for the specific model.

        Returns:
            None
        """
        print(f"Running model: {self.model_name} on image: {image_path}")
        try:
            if self.model_name in [
                "kosmos2",
                "rtdetr",
                "llava",
                "groundingdino",
                "yolox",
                "owlvit",
                "yolov8",
                "yolov5",
                "yolov6",
                "yolov7",
                "yolov11",
                "yolov10",
                "detectron2",
                "detr",
                "detrclip",
                "sam2",
            ]:
                self._run_model(image_path, **kwargs)
            else:
                raise ValueError("Invalid model selected.")
        except Exception as e:
            print(f"Error occurred: {e}")

    def _run_model(self, image_path, **kwargs):
        if self.model_name == "kosmos2":
            objects = self.model.detect_objects(image_path)
            self.model.visualize_results(image_path, objects)
        elif self.model_name == "rtdetr":
            image, results = self.model.detect_objects(image_path, **kwargs)
            self.model.visualize_results(image, results)
        elif self.model_name == "llava":
            question = kwargs.get("question", "What is shown in this image?")
            objects, text_output = self.model.detect_objects(
                image_path, question=question
            )
            self.model.visualize_results(image_path, objects, text_output)
        elif self.model_name == "groundingdino":
            text_prompt = kwargs.get("text_prompt", "object")
            result = self.model.detect_objects(image_path, text_prompt)
            self.model.visualize_results(result)
        elif self.model_name == "yolox":
            conf_thre = kwargs.get("conf_thre", 0.25)
            nms_thre = kwargs.get("nms_thre", 0.45)
            image, outputs, scale = self.model.detect_objects(
                image_path, conf_thre=conf_thre, nms_thre=nms_thre
            )
            self.model.visualize_results(image, outputs, scale)
        elif self.model_name == "owlvit":
            queries = kwargs.get(
                "queries", ["a cat", "a dog", "a person", "a car", "a tree"]
            )
            image, results = self.model.detect_objects(image_path, queries)
            self.model.visualize_results(image, results, queries)
        elif self.model_name == "yolov8":
            results = self.model.detect_objects(image_path)
            self.model.visualize_results(results)
        elif self.model_name == "yolov5":
            results, img_bgr = self.model.detect_objects(image_path)
            self.model.visualize_results(results, img_bgr)
        elif self.model_name == "yolov6":
            output_path = self.model.detect_objects(
                image_path, threshold=kwargs.get("threshold", 0.25)
            )
            self.model.visualize_results(output_path)
        elif self.model_name == "yolov7":
            output_path = self.model.detect_objects(
                image_path, threshold=kwargs.get("threshold", 0.25)
            )
            self.model.visualize_results(output_path)
        elif self.model_name == "yolov11":
            result_image_path = self.model.detect_objects(
                image_path, conf_threshold=kwargs.get("conf_threshold", 0.25)
            )
            self.model.visualize_results(result_image_path)
        elif self.model_name == "yolov10":
            result_image_path = self.model.detect_objects(
                image_path, conf_threshold=kwargs.get("conf_threshold", 0.25)
            )
            self.model.visualize_results(result_image_path)
        elif self.model_name == "detectron2":
            outputs = self.model.detect_objects(image_path)
            self.model.visualize_results(outputs)
        elif self.model_name == "detr":
            image = self.model.load_image(image_path)
            results = self.model.detect_objects(
                image, score_threshold=kwargs.get("score_threshold", 0.7)
            )
            self.model.visualize_results(image, results)
        elif self.model_name == "detrclip":
            image = self.model.load_image(image_path)
            results = self.model.detect_objects(
                image, threshold=kwargs.get("threshold", 0.5)
            )
            self.model.visualize_results(image, results)
        elif self.model_name == "sam2":
            image_bgr, detections = self.model.detect_objects(image_path)
            self.model.visualize_results(image_bgr, detections)
