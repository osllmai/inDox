from tenacity import retry, stop_after_attempt, wait_random_exponential


class OpenAi:
    def __init__(self, api_key, model, base_url=None):
        """
        Initializes the GPT-3 model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT-3 model version.
        """
        from openai import OpenAI

        try:
            self.model = model
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise Exception(f"{e}")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages, max_tokens, temperature, frequency_penalty, presence_penalty, top_p, stream):
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model.
            max_tokens (int): The maximum number of tokens in the generated response.
            temperature (float): The sampling temperature.
            frequency_penalty (float): The frequency penalty.
            presence_penalty (float): The presence penalty.
            top_p (float): The top_p parameter for nucleus sampling.
            stream: Indicates if the response should be streamed.

        Returns:
            str: The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                stream=stream
            )

            if stream:
                # If streaming, accumulate the response content
                result = ""
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, 'content', None)
                    if content is not None:
                        result += content
                result = result.strip()
            else:
                # For non-streaming response
                result = response.choices[0].message.content.strip()

            return result

        except Exception as e:
            raise Exception(f"{e}")

    def chat(self, prompt, system_prompt="You are a helpful assistant", max_tokens=None, temperature=0.2,
             frequency_penalty=None, presence_penalty=None, top_p=None, stream=None):
        """
        Public method to interact with the model using chat messages.

        Args:
            prompt (str): The prompt to generate a response for.
            system_prompt (str): The system prompt.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to None.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The nucleus sampling parameter.
            stream

        Returns:
            str: The generated response.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            stream=stream
        )
