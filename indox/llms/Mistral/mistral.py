import logging
import requests

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class MistralQA:
    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", prompt_template=None):
        """
        Initializes the Mistral 7B Instruct model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The Mistral model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        try:
            logging.info("Initializing MistralQA with model: %s", model)
            self.model = model
            self.api_key = api_key
            self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logging.info("MistralQA initialized successfully")
        except ValueError as ve:
            logging.error("ValueError during initialization: %s", ve)
            raise
        except Exception as e:
            logging.error("Unexpected error during initialization: %s", e)
            raise

    def _send_request(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": prompt
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
                return generated_text
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logging.error(error_message)
                raise Exception(error_message)
        except Exception as e:
            logging.error("Error in _send_request: %s", e)
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
        prompt = self.prompt_template.format(context=context, question=question)
        return self._send_request(prompt)

    def answer_question(self, context, question, prompt_template=None):
        """
        Answer a question based on the given context using the Mistral 7B Instruct model.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
            prompt_template (str, optional): The template for the prompt. Defaults to None.

        Returns:
            str: The generated answer.
        """
        try:
            logging.info("Answering question: %s", question)
            self.prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(context, question)
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
            return self._send_request(prompt)
        except Exception as e:
            logging.error("Error in get_summary: %s", e)
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
            return self._send_request(prompt)
        except Exception as e:
            logging.error("Error in answer_with_agent: %s", e)
            return str(e)
