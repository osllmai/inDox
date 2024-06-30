import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


class GoogleAi:
    def __init__(self, api_key, model="gemini-1.5-flash-latest"):
        """
        Initializes with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for Google Ai.
            model (str): The Gemini model version.
        """
        import google.generativeai as genai
        try:
            logging.info("Initializing GoogleAi with model: %s", model)
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logging.info("GoogleAi initialized successfully")
        except Exception as e:
            logging.error("Error initializing GoogleAi: %s", e)
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
            logging.info("Generating response")
            response = self.model.generate_content(contents=prompt)
            logging.info("Response generated successfully")
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logging.error("Error generating response: %s", e)
            raise

    def _format_prompt(self, context, question, max_tokens):
        """
        Formats the prompt for generating a response.

        Args:
            context (str): The context for the prompt.
            question (str): The question for the prompt.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The formatted prompt.
        """
        return f"Given Context: {context} Give the best full answer amongst the option to question {question} in maximum {max_tokens} tokens"

    def answer_question(self, context, question, max_tokens=200):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated answer. Defaults to 200.

        Returns:
            str: The generated answer.
        """
        try:
            logging.info("Answering question")
            prompt = self._format_prompt(context, question, max_tokens)
            return self._generate_response(prompt)
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
        prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation} in maximum 150 tokens"
        try:
            logging.info("Generating summary for documentation")
            return self._generate_response(prompt)
        except Exception as e:
            logging.error("Error generating summary: %s", e)
            return str(e)

    def grade_docs(self, context, question):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
        """
        filtered_docs = []
        system_prompt = """
            You are a grader assessing the relevance of a retrieved document to a user question.
            If the document contains any keywords or evidence related to the user question, even if minimal, 
            grade it as relevant. The goal is to filter out only completely erroneous retrievals.
            Give a binary "yes" or "no" score to indicate whether the document is relevant to the question.

            Provide the score with no preamble or explanation.
        """
        for doc in context:
            prompt = f"Here is the retrieved document:\n{doc}\nHere is the user question:\n{question}"
            try:
                grade = self._generate_response(prompt).lower()
                if grade == "yes":
                    logging.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logging.info("Not relevant doc")
            except Exception as e:
                logging.error("Error grading document: %s", e)
        return filtered_docs

    def check_hallucination(self, context, answer):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The context for checking.
            answer (str): The answer to check.

        Returns:
            str: 'yes' if the answer is grounded, 'no' otherwise.
        """
        system_prompt = """
            You are a grader assessing whether an answer is grounded in / supported by a set of facts.
            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded / supported by the set of facts.
            Provide the score with no preamble or explanation.
        """
        prompt = f"Here are the facts:\n{context}\nHere is the answer:\n{answer}"
        try:
            logging.info("Checking hallucination for answer")
            return self._generate_response(prompt).lower()
        except Exception as e:
            logging.error("Error checking hallucination: %s", e)
            return str(e)
