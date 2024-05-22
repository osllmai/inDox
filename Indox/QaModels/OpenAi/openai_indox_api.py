import requests


class OpenAiQaIndoxApi:
    def __init__(self, api_key):
        self.api_key = api_key

    def _attempt_answer_question(self, context, question):
        """
        Generates an answer to the given question using the Mistral model via the Inference API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": 150,
            "messages": [
                {
                    "content": "you are a helpful assistant.",
                    "role": "system"
                },
                {
                    "content": f"Context: {context}\nQuestion: {question}\nAnswer:",
                    "role": "user"
                }
            ],
            "model": "gpt-3.5-turbo-0125",
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.3,
            "top_p": 1
        }

        # Send the request to the Indox API
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # If the response is a list, extract the first item
            answer_data = response.json()
            generated_text = answer_data.get("total_token_length", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")

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
