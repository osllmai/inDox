import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates


class LLaVANextObjectDetector:
    def __init__(self, model_id="lmms-lab/llama3-llava-next-8b", device="cuda"):
        # Initialize the model and tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load the pretrained LLaVA model and tokenizer
        model_name = "llava_llama3"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name=model_name,
            device_map="auto",
            load_8bit=True,
            attn_implementation=None
        )

        self.model.eval()
        self.model.tie_weights()
        self.device = device

    def detect_objects(self, image_path, question="What is shown in this image?"):
        """
        Detect objects in an image.

        Args:
            image_path (str): Path to the input image.
            question (str): A question or prompt to guide the detection.

        Returns:
            tuple: A list of detected objects and the raw text output from the model.
        """
        # Load image from the given path
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Prepare inputs with prompt for grounding
        conv_template = "llava_llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Tokenize the input and prepare the image tensor
        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [img.to(dtype=torch.float16, device=self.device) for img in image_tensor]
        image_sizes = [image.size]

        # Generate the response from LLaVA
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
            tokenizer=self.tokenizer
        )

        # Decode the output text from LLaVA
        text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

        # Post-process the output to extract entities like objects and their possible bounding boxes
        objects = self.extract_objects_from_text(text_output)

        return objects, text_output

    def extract_objects_from_text(self, text_output):
        """
        Extract objects and bounding boxes from the model's text output.

        Args:
            text_output (str): The text output from the model.

        Returns:
            list: A list of detected objects with names and bounding boxes.
        """
        objects = []
        lines = text_output.split("\n")
        for line in lines:
            # Example structure: "dog: bounding box (0.1, 0.2, 0.5, 0.5)"
            if "bounding box" in line:
                parts = line.split(":")
                object_name = parts[0].strip()
                bbox_str = parts[1].strip().replace("(", "").replace(")", "")
                bbox = tuple(map(float, bbox_str.split(", ")))
                objects.append([object_name, bbox])  # Store object name and bounding box
        return objects

    def visualize_results(self, image_path, objects, text_output):
        """
        Visualize detected objects in an image.

        Args:
            image_path (str): Path to the input image.
            objects (list): List of detected objects with bounding boxes.
            text_output (str): The raw text output from the model.

        Returns:
            numpy.ndarray: The annotated image.
        """
        # Load the image using OpenCV
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Convert bounding boxes from normalized coordinates to pixel values
        h, w, _ = cv_image.shape
        for entity_name, bbox_list in objects:
            bbox = bbox_list[0]  # Access the tuple inside the list
            x_min, y_min, x_max, y_max = bbox  # Assuming bbox is a 4-element tuple

            # Convert to pixel coordinates
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)

            # Draw the bounding box on the image
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(cv_image, entity_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert from BGR to RGB for display with Matplotlib
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axis
        plt.title(text_output, fontsize=12)
        plt.show()

        return cv_image
