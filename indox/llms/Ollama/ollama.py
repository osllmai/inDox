import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class Ollama:
    def __init__(self, model):
        """
        Initializes the Ollama model with the specified model version.

        Args:
            model (str): The Ollama model version.
        """

        try:
            logging.info(f"Initializing Ollama with model: {model}")
            self.model = model
            logging.info("Ollama initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Ollama: {e}")
            raise

    def run_ollama_command(self, command):
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error("Error running Ollama command")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question):
        """
        Generates an answer to the given question using the Ollama model.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.


        Returns:
            str: The generated answer.
        """
        prompt = f"Given Context: {context} Give the best full answer amongst the option to question {question}"

        try:
            logging.info("Attempting to generate an answer")
            command = f"ollama run {self.model} --nowordwrap [{prompt}]"
            response = self.run_ollama_command(command)
            logging.info("Answer generated successfully")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.


        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Answering question")
            return self._attempt_answer_question(context, question)
        except Exception as e:
            logging.error(f"Error in answer_question: {e}")
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
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation}"
            command = f"ollama run {self.model} --nowordwrap [{prompt}]"
            response = self.run_ollama_command(command)
            logging.info("Summary generated successfully")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return str(e)

    def grade_docs(self, context, question):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
        """
        filtered_docs = []
        try:
            system_prompt = f"""
           You are a grader assessing the relevance of a retrieved document to a user question.
            If the document contains any keywords or evidence related to the user question, even if minimal, 
            grade it as relevant. The goal is to filter out only completely erroneous retrievals.
            Give a binary "yes" or "no" score to indicate whether the document is relevant to the question.

            Provide the score with no preamble or explanation.
            """
            for i in range(len(context)):
                prompt = f"""
                    Here is the retrieved document:
                    {context[i]}
                    Here is the user question: 
                    {question}"""
                command = f"ollama run {self.model} --nowordwrap [{prompt}]"
                response = self.run_ollama_command(command)
                grade = response.strip()
                if grade.lower() == "yes":
                    logging.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif grade.lower() == "no":
                    logging.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logging.error(f"Error generating agent answer: {e}")
            return str(e)

    def check_hallucination(self, context, answer):
        try:
            system_prompt = """
                You are a grader assessing whether an answer is grounded in / supported by a set of facts.
                Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded / supported by a set
                 of facts. Provide score with no preamble or explanation.
                """
            prompt = f"""
                Here are the facts:
                \n -------- \n
                {context}
                \n -------- \n
                Here is the answer : {answer}
                """

            command = f"ollama run {self.model} --nowordwrap [{prompt}]"
            response = self.run_ollama_command(command)
            hallucination_answer = response.strip()
            logging.info("Agent answer generated successfully")
            return hallucination_answer
        except Exception as e:
            logging.error(f"Error generating agent answer: {e}")
            return str(e)
