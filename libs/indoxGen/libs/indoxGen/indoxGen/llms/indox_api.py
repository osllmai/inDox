import requests


def format_prompt(context, question):
    """
    Formats the prompt for generating a response.

    Args:
        context (str): The context for the prompt.
        question (str): The question for the prompt.

    Returns:
        str: The formatted prompt.
    """
    return f"Given Context: {context} Give the best full answer amongst the option to question {question}"


class IndoxApi:
    api_key: str

    def __init__(self, api_key, prompt_template=""):
        """
        Initializes the IndoxApi with the specified API key and an optional prompt template.

        Args:
            api_key (str): The API key for Indox API.
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        self.api_key = api_key
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, system_prompt, user_prompt, max_tokens, temperature, stream, model,
                      presence_penalty, frequency_penalty, top_p):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": user_prompt,
                    "role": "user"
                }
            ],
            "model": model,
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")

    def _attempt_answer_question(self, context, question, max_tokens, temperature, stream, model,
                                 presence_penalty, frequency_penalty, top_p):
        """
        Generates an answer to a question based on the given context using the Indox API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """
        system_prompt = "You are a helpful assistant."
        user_prompt = self.prompt_template.format(context=context, question=question, )
        return self._send_request(system_prompt, user_prompt, max_tokens=max_tokens,
                                  temperature=temperature, stream=stream,
                                  model=model, presence_penalty=presence_penalty,
                                  frequency_penalty=frequency_penalty,
                                  top_p=top_p)

    def chat(self, prompt, system_prompt="You are a helpful assistant", max_tokens=16384, temperature=0.3, stream=True,
             model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
        return self._send_request(system_prompt=system_prompt, user_prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, stream=stream,
                                  model=model, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                                  top_p=top_p)
