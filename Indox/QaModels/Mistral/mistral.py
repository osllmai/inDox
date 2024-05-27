from typing import List

import requests
from pydantic import BaseModel


class MistralQA:
    def __init__(self, api_key, model):
        self.model = model
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("A valid Hugging Face API key is required.")

    def _attempt_answer_question(self, context, question):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": f"Context: {context}\nQuestion: {question}\nAnswer:"
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            answer_data = response.json()
            if isinstance(answer_data, list) and len(answer_data) > 0:
                answer_data = answer_data[0]

            generated_text = answer_data.get("generated_text", "")

            answer_split = generated_text.split("Answer:", 1)
            if len(answer_split) > 1:
                return answer_split[1].strip().split("\n")[0]

            return "No answer found."
        else:
            raise Exception(f"Error from Hugging Face API: {response.status_code}, {response.text}")

    def answer_question(self, context, question):
        try:
            return self._attempt_answer_question(context, question)
        except Exception as e:
            print(e)
            return str(e)


class MistralAgent(BaseModel):
    model: str
    api_key: str

    def generate(self, prompt: str, stop: List[str] = None):

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers=headers,
            json={"inputs": f"Context: \nQuestion: {prompt}\n Answer:"}
        )
        if response.status_code == 200:
            answer_data = response.json()
            if isinstance(answer_data, list) and len(answer_data) > 0:
                answer_data = answer_data[0]

            generated_text = answer_data.get("generated_text", "")

            answer_split = generated_text.split("Answer:", 1)
            if len(answer_split) > 1:
                return answer_split[1].strip().split("\n")[0], generated_text

            return "No answer found."
        else:
            raise Exception(f"Error from Hugging Face API: {response.status_code}, {response.text}")


