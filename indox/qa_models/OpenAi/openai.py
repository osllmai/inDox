import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import os

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
class OpenAiQA:
    def __init__(self, api_key, model):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT-3 model version.
        """
        try:
            logging.info("Initializing OpenAiQA with model: %s", model)
            self.model = model
            self.client = OpenAI(api_key=api_key)
            logging.info("OpenAiQA initialized successfully")
        except Exception as e:
            logging.error("Error initializing OpenAiQA: %s", e)
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None, temperature=0):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Attempting to generate an answer for the question: %s", question)
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
                max_tokens=max_tokens,
                stop=stop_sequence
            )

            answer = response.choices[0].message.content.strip()
            logging.info("Answer generated successfully")
            return answer
        except Exception as e:
            logging.error("Error generating answer: %s", e)
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Answering question: %s", question)
            return self._attempt_answer_question(
                context,
                question,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=0,
            )
        except Exception as e:
            logging.error("Error in answer_question: %s", e)
            return str(e)
