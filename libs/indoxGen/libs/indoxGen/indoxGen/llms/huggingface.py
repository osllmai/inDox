import requests


class HuggingFaceModel:
    api_key: str
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    prompt_template: str = ""

    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", prompt_template=""):
        """
        Initializes the specified model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        try:
            self.model = model
            self.api_key = api_key
            self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
        except ValueError as ve:
            raise
        except Exception as e:
            raise

    def _send_request(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": prompt,
        }

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]
                generated_text = answer_data.get("generated_text", "")
                return generated_text
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                raise Exception(error_message)
        except Exception as e:
            raise

    def chat(self, prompt, system_prompt=""):
        """
        A method to send a combined system and user prompt to the model for generating or judging synthetic data.
        """
        combined_prompt = system_prompt + "\n" + prompt
        try:
            response = self._send_request(prompt=combined_prompt)
            return response.strip()
        except Exception as e:
            raise
