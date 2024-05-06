import os
from openai import OpenAI
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .utils import read_config
import requests


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo-0125"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.config = read_config()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
            self, context, question, max_tokens=150, stop_sequence=None, temperature=0
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context,
                question,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=self.config["qa_model"]["temperature"],
            )
        except Exception as e:
            print(e)
            return e


class Mistral7BQAModel(BaseQAModel):
    def __init__(self, model="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the Mistral 7B Instruct model via the Hugging Face Inference API.

        Args:
            model (str, optional): The Mistral model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        """
        self.model = model
        self.api_key = os.getenv("HF_API_KEY")  # Ensure the API key is set in the environment variables
        self.config = read_config()
        if not self.api_key:
            raise ValueError("A valid Hugging Face API key is required.")

    def _attempt_answer_question(self, context, question):
        """
        Generates an answer to the given question using the Mistral model via the Inference API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        # The payload should only include the question and context in a dictionary
        payload = {
            "inputs": f"Context: {context}\nQuestion: {question}\nAnswer:"
        }

        # Send the request to the Hugging Face API
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            # If the response is a list, extract the first item
            answer_data = response.json()
            if isinstance(answer_data, list) and len(answer_data) > 0:
                answer_data = answer_data[0]

            generated_text = answer_data.get("generated_text", "")

            # Extract the text following "Answer:"
            answer_split = generated_text.split("Answer:", 1)
            if len(answer_split) > 1:
                return answer_split[1].strip().split("\n")[0]  # Extract only the first line after "Answer:"

            return "No answer found."
        else:
            raise Exception(f"Error from Hugging Face API: {response.status_code}, {response.text}")

    def answer_question(self, context, question):
        """
        Answer a question based on the given context using the Mistral 7B Instruct model.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.

        Returns:
            str: The generated answer.
        """
        try:
            return self._attempt_answer_question(context, question)
        except Exception as e:
            print(e)
            return str(e)


def choose_qa_model():
    config = read_config()
    model_name = config["qa_model"]["name"].lower()
    if model_name == "openai" or "":
        return GPT3TurboQAModel()
    elif model_name == "mistral":
        return Mistral7BQAModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
