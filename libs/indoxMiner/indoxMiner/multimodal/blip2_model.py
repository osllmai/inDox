from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os


class BLIP2:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda", use_8bit=True):
        """
        Initializes the BLIP-2 model, processor, and device settings.

        Args:
            model_name (str): Hugging Face model name.
            device (str): Device to load the model on ("cuda" or "cpu").
            use_8bit (bool): Whether to load model in 8-bit for lower memory usage.
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load the processor
        self.processor = Blip2Processor.from_pretrained(model_name)

        # Load the model with optional 8-bit quantization
        model_kwargs = {"torch_dtype": torch.float16, "device_map": {"": 0}}


        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)  # Ensure model is on the correct device

    def process_image(self, image_path: str):
        """
        Loads and processes a local image.

        Args:
            image_path (str): Local path to the image.

        Returns:
            torch.Tensor: Processed image tensor.
            tuple: Image size.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The image path '{image_path}' does not exist.")

        image = Image.open(image_path).convert("RGB")
        return image, image.size

    def generate_response(self, image_path: str, question: str = None):
        """
        Generates a response from the model given an image and an optional question.

        If no question is provided, the model will generate a caption.

        Args:
            image_path (str): Local path to the image.
            question (str, optional): Text prompt for the model. Defaults to None.

        Returns:
            str: Generated response.
        """
        image, image_size = self.process_image(image_path)

        if question:
            prompt = f"Question: {question} Answer:"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
