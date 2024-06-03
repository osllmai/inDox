import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import os

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class OpenAiQA:
    def __init__(self, api_key, model, prompt_template=None):
        """
        Initializes the GPT-3 model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT-3 model version.
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        try:
            logging.info("Initializing OpenAiQA with model: %s", model)
            self.model = model
            self.client = OpenAI(api_key=api_key)
            self.prompt_template = prompt_template or "Given Context: {context} Give the best full answer amongst the option to question {question}"
            logging.info("OpenAiQA initialized successfully")
        except Exception as e:
            logging.error("Error initializing OpenAiQA: %s", e)
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None, temperature=0):
        """
        Generates an answer to the given question using the GPT-3 model.

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
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Question Answering Portal"},
                    {"role": "user", "content": prompt},
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
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None, prompt_template=None):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.
            prompt_template (str, optional): The template for the prompt. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Answering question: %s", question)
            prompt_template = prompt_template or self.prompt_template
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

    def get_summary(self, documentation):
        """
        Generates a detailed summary of the provided documentation.

        Args:
            documentation (str): The documentation to summarize.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Generating summary for documentation")
            prompt = "You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n" + documentation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
            )
            summary = response.choices[0].message.content.strip()
            logging.info("Summary generated successfully")
            return summary
        except Exception as e:
            logging.error("Error generating summary: %s", e)
            return str(e)

    def answer_with_agent(self, context, question, tool_description, tool_names):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
            tool_description (str): Description of the available tools.
            tool_names (str): The names of the available tools.

        Returns:
            str: The generated answer.
        """
        try:
            logging.info("Answering question with agent: %s", question)
            system_prompt = f"""
            Answer the following questions and obey the following commands as best you can.

            You have access to the following tools:

            {tool_description}

            You will receive a message from the human, then you should start a loop and do one of two things

            Option 1: You use a tool to answer the question.
            For this, you should use the following format:
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: "the input to the action, to be sent to the tool"

            After this, the human will respond with an observation, and you will continue.

            Option 2: You respond to the human.
            For this, you should use the following format:
            Action: Response To Human
            Action Input: "your response to the human, summarizing what you did and what you learned"

            Begin!
            """
            prompt = f"{system_prompt}\nContext: {context}\nQuestion: {question}\n"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
            )
            agent_answer = response.choices[0].message.content.strip()
            logging.info("Agent answer generated successfully")
            return agent_answer
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
            return str(e)
