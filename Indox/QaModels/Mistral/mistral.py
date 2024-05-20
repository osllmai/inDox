import requests


class MistralQA:
    def __init__(self, api_key, model):
        """
        Initializes the Mistral 7B Instruct model via the Hugging Face Inference API.

        Args:
            model (str, optional): The Mistral model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        """
        self.model = model
        self.api_key = api_key
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
