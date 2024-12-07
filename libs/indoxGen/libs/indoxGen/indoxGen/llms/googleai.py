from tenacity import retry, stop_after_attempt, wait_random_exponential


class GoogleAi:
    def __init__(self, api_key, model="gemini-1.5-flash-latest"):
        """
        Initializes the Google AI model with the specified model version.

        Args:
            api_key (str): The API key for Google AI.
            model (str): The Google AI model version. Defaults to "gemini-1.5-flash-latest".
        """
        import google.generativeai as genai  # Assuming this is the Google API module for Gemini

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except Exception as e:
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, prompt):
        """
        Generates a response using the model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response text.
        """
        try:
            response = self.model.generate_content(contents=prompt)
            return response.text.strip().replace("\n", "")
        except Exception as e:
            raise

    def chat(self, prompt, system_prompt=""):
        """
        A method to send a combined system and user prompt to the Google AI API for data generation or judging.

        Args:
            prompt (str): The user prompt to be sent to the model.
            system_prompt (str, optional): An optional system prompt for guiding the model. Defaults to "".

        Returns:
            str: The generated response from the model.
        """
        combined_prompt = system_prompt + "\n" + prompt
        try:
            return self._generate_response(combined_prompt)
        except Exception as e:
            return str(e)
