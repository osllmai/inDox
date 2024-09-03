import requests
from loguru import logger
import sys
from indox.core import BaseLLM

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class HuggingFaceModel(BaseLLM):
    api_key: str
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    prompt_template: str = ""

    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", prompt_template=""):
        super().__init__(api_key=api_key, model=model, prompt_template=prompt_template)
        """
        Initializes the specified model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        try:
            logger.info(f"Initializing HuggingFaceModel with model: {model}")
            self.model = model
            self.api_key = api_key
            self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logger.info("HuggingFaceModel initialized successfully")
        except ValueError as ve:
            logger.error(f"ValueError during initialization: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _send_request(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": prompt,
        }

        try:
            logger.info("Sending request to Hugging Face API")
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                logger.info("Received successful response from Hugging Face API")
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]
                generated_text = answer_data.get("generated_text", "")
                # Extract only the answer part
                return generated_text
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
        except Exception as e:
            logger.error(f"Error in _send_request: {e}")
            raise

    def _attempt_answer_question(self, context, question):
        """
        Generates an answer to the given question using the specified model via the Inference API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """
        prompt = self.prompt_template.format(context=context, question=question)
        response = self._send_request(prompt)
        answer = response.split("Answer:")[-1].strip()

        return answer

    def answer_question(self, context, question, prompt_template=None):
        """
        Answer a question based on the given context using the specified model.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
            prompt_template (str, optional): The template for the prompt. Defaults to None.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Answering question")
            self.prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(context, question)
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
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
            logger.info("Generating summary for documentation")
            prompt = "You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n" + documentation
            return self._send_request(prompt)
        except Exception as e:
            logger.error(f"Error in get_summary: {e}")
            return str(e)

    def grade_docs(self, context, question):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (list): A list of documents to grade.
            question (str): The question to answer.

        Returns:
            list: Filtered list of relevant documents.
        """
        filtered_docs = []
        try:
            system_prompt = """
            You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keywords related to the user question, grade it as relevant. 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the score with no preamble or explanation. 
            """
            for i in range(len(context)):
                user_prompt = f"""
                Here is the retrieved document:
                {context[i]}
                Here is the user question:
                {question}
                """
                response = self._send_request(system_prompt + "\n" + user_prompt)
                response = response.split("\n")[-1]
                if response.lower().strip() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif response.lower().strip() == "no":
                    logger.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def check_hallucination(self, context, answer):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The text to base the answer on.
            answer (str): The answer to check for hallucination.

        Returns:
            str: 'yes' if the answer is grounded, 'no' otherwise.
        """
        try:
            system_prompt = """
            You are a grader assessing whether an answer is grounded in 
            / supported by a set of facts. 
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded 
            / supported by a set of facts. Provide score with no preamble or explanation.
            """
            user_prompt = f"""
            Here are the facts:
            \n -------- \n
            {context}
            \n -------- \n
            Here is the answer: {answer}
            """
            response = self._send_request(system_prompt + "\n" + user_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def chat(self, prompt):
        return self._send_request(prompt=prompt)
