from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import os


class LLaVA:
    def __init__(self, pretrained="lmms-lab/llama3-llava-next-8b", model_name="llava_llama3", device="cuda", device_map="auto"):
        """
        Initializes the LLaVA model, tokenizer, and image processor.

        Args:
            pretrained (str): Path or Hugging Face model name.
            model_name (str): Model identifier.
            device (str): Device to load the model on ("cuda" or "cpu").
            device_map (str): Device map for model loading.
        """
        self.device = device
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, attn_implementation="eager"
        )
        self.model.to(self.device)

        self.model.eval()

    def process_image(self, image_path: str):
        """
        Loads and processes a local image for model inference.

        Args:
            image_path (str): Local path to the image.

        Returns:
            torch.Tensor: Processed image tensor.
            tuple: Image size.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The image path '{image_path}' does not exist.")

        image = Image.open(image_path).convert("RGB")

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        return image_tensor, image.size

    def generate_response(self, image_path: str, question: str):
        """
        Generates a response from the model given an image and a text prompt.

        Args:
            image_path (str): Local path to the image.
            question (str): Text prompt for the model.

        Returns:
            str: Generated response.
        """
        image_tensor, image_size = self.process_image(image_path)

        conv_template = "llava_llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
