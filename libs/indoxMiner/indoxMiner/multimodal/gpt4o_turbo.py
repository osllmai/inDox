import os
import base64
from openai import OpenAI
from PIL import Image
import io


class GPT4o:
    def __init__(self, api_key: str = None):
        """
        Initializes the GPT-4 Turbo model for vision-language tasks.

        Args:
            api_key (str): OpenAI API key. If not provided, it must be set as an environment variable `OPENAI_API_KEY`.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it explicitly.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def encode_image(self, image_path: str):
        """
        Converts a local image to Base64 encoding for OpenAI API.

        Args:
            image_path (str): Path to the local image.

        Returns:
            str: Base64-encoded image string.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The image path '{image_path}' does not exist.")

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

    def generate_response(self, image_path: str, question: str):
        """
        Generates a response from GPT-4 Turbo based on a local image and a text question.

        Args:
            image_path (str): Path to the local image.
            question (str): Question related to the image.

        Returns:
            str: Model-generated response.
        """
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()
