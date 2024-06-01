import logging
import requests

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class MistralQA:
    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the Mistral 7B Instruct model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The Mistral model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        """
        try:
            logging.info("Initializing MistralQA with model: %s", model)
            self.model = model
            self.api_key = api_key
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logging.info("MistralQA initialized successfully")
        except ValueError as ve:
            logging.error("ValueError during initialization: %s", ve)
            raise
        except Exception as e:
            logging.error("Unexpected error during initialization: %s", e)
            raise

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
        payload = {
            "inputs": f"Context: {context}\nQuestion: {question}\nAnswer:"
        }

        try:
            logging.info("Sending request to Hugging Face API")
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                logging.info("Received successful response from Hugging Face API")
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]

                generated_text = answer_data.get("generated_text", "")

                answer_split = generated_text.split("Answer:", 1)
                if len(answer_split) > 1:
                    return answer_split[1].strip().split("\n")[0]

                return "No answer found."
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logging.error(error_message)
                raise Exception(error_message)
        except Exception as e:
            logging.error("Error in _attempt_answer_question: %s", e)
            raise

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
            logging.info("Answering question: %s", question)
            return self._attempt_answer_question(context, question)
        except Exception as e:
            logging.error("Error in answer_question: %s", e)
            return str(e)
