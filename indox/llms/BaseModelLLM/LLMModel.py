import abc
import sys
import requests
import ollama as ol
from abc import ABC
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential


# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class LLMModel(BaseModel , ABC):

    @abc.abstractmethod
    def answer_question(self, context, question, max_tokens=200):
        pass

    @abc.abstractmethod
    def get_summary(self, documentation):
        pass

    @abc.abstractmethod
    def grade_docs(self, context, question):
        pass

    @abc.abstractmethod
    def check_hallucination(self, context, answer):
        pass

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
            logger.info(f"Initializing GoogleAi with model: {model}")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("GoogleAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GoogleAi: {e}")
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
            logger.info("Generating response")
            response = self.model.generate_content(contents=prompt)
            logger.info("Response in generated successfully")
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
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
            logger.info("Answering question")
            prompt = self._format_prompt(context, question, max_tokens)
            return self._generate_response(prompt)
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
        prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation} in maximum 150 tokens"
        try:
            logger.info("Generating summary for documentation")
            return self._generate_response(prompt)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
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
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logger.info("Not relevant doc")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
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
            logger.info("Checking hallucination for answer")
            return self._generate_response(prompt).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)

class HuggingFaceModel:
    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", prompt_template=None):
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

class IndoxApi:
    def __init__(self, api_key, prompt_template=None):
        """
        Initializes the IndoxApiOpenAiQa with the specified API key and an optional prompt template.

        Args:
            api_key (str): The API key for Indox API.
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        self.api_key = api_key
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, system_prompt, user_prompt):
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
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": user_prompt,
                    "role": "user"
                }
            ],
            "model": "gpt-3.5-turbo-0125",
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.3,
            "top_p": 1
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")

    def _attempt_answer_question(self, context, question):
        """
        Generates an answer to a question based on the given context using the Indox API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """
        system_prompt = "You are a helpful assistant."
        user_prompt = self.prompt_template.format(context=context, question=question)
        return self._send_request(system_prompt, user_prompt)

    def answer_question(self, context, question, prompt_template=None):
        """
        Answer a question based on the given context using the Indox API.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
            prompt_template (str, optional): The template for the prompt. Defaults to None.

        Returns:
            str: The generated answer.
        """
        try:
            # prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(context, question)
        except Exception as e:
            logger.error(e)
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
            system_prompt = "You are a helpful assistant."
            user_prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation}"
            return self._send_request(system_prompt, user_prompt)
        except Exception as e:
            logger.error(e)
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
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
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
                response = self._send_request(system_prompt, user_prompt)
                if response.lower() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif response.lower() == "no":
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
            You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded / supported by a set of facts. Provide score with no preamble or explanation.
            """
            user_prompt = f"""
            Here are the facts:
            \n -------- \n
            {context}
            \n -------- \n
            Here is the answer: {answer}
            """
            response = self._send_request(system_prompt, user_prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def assistant(self, system_prompt, user_prompt, role, history):
        try:
            response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt, role=role, history=history)
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

class Mistral:
    def __init__(self, api_key, model="mistral-medium-latest"):
        """
        Initializes the Mistral AI model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for Mistral AI.
            model (str): The Mistral AI model version.
        """
        from mistralai.client import MistralClient
        try:
            logger.info(f"Initializing MistralAI with model: {model}")
            self.model = model
            self.client = MistralClient(api_key=api_key)
            logger.info("MistralAI initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MistralAI: {e}")
            raise

    def run_mistral(self, user_message):
        """
        Runs the Mistral model to generate a response based on the user message.

        Args:
            user_message (str): The message to be processed by the Mistral model.

        Returns:
            str: The generated response.
        """
        from mistralai.models.chat_completion import ChatMessage

        try:
            messages = [
                ChatMessage(role="user", content=user_message)
            ]
            chat_response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in run_mistral: {e}")
            return str(e)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None, temperature=0):
        """
        Generates an answer to the given question using the Mistral AI model.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated summary.
        """
        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """

        try:
            logger.info("Attempting to generate an answer for the question")
            return self.run_mistral(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
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
            logger.info("Answering question")
            return self._attempt_answer_question(
                context,
                question,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=0,
            )
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
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided in maximum 100 token.\n\nDocumentation:\n{documentation}"
            return self.run_mistral(prompt)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
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
            You are a grader assessing relevance of a retrieved
            document to a user question. If the document contains keywords related to the
            user question, grade it as relevant. It does not need to be a stringent test.
            The goal is to filter out erroneous retrievals.

            Give a binary score 'yes' or 'no' score to indicate whether the document is
            relevant to the question.

            Provide the score with no preamble or explanation.
            """
            for i in range(len(context)):
                prompt = f"""
                    Here is the retrieved document:
                    {context[i]}
                    Here is the user question:
                    {question}"""
                grade = self.run_mistral(system_prompt + prompt).strip()
                if grade.lower() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif grade.lower() == "no":
                    logger.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
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
            return self.run_mistral(system_prompt + prompt).strip()
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

class Ollama:
    def __init__(self, model):
        """
        Initializes the Ollama model with the specified model version and an optional prompt template.

        Args:
            model (str): Ollama model version.
        """

        try:
            logger.info(f"Initializing Ollama with model: {model}")
            self.model = model
            logger.info("Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages, max_tokens=250, temperature=0):
        """
        Generates a response from the Ollama model.

        Args:
            messages : The  messages to send to the model.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response")
            response = ol.generate(model=self.model, prompt=messages)
            result = response["response"].strip().replace("\n", "").replace("\t", "")
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_prompt(self, context, question):
        """
        Formats the prompt for generating a response.

        Args:
            context (str): The context for the prompt.
            question (str): The question for the prompt.

        Returns:
            str: The formatted prompt.
        """
        return f"Given Context: {context} Give the best full answer amongst the option to question {question}"

    def answer_question(self, context, question, max_tokens=350):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 350.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Answering question")
            prompt = self._format_prompt(context, question)
            return self._generate_response(messages=prompt, max_tokens=max_tokens, temperature=0)
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
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            messages = prompt
            return self._generate_response(messages, max_tokens=150, temperature=0)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
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
            messages = system_prompt + prompt
            try:
                grade = self._generate_response(messages, max_tokens=150, temperature=0).lower()
                if grade == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logger.info("Not relevant doc")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
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
        messages = system_prompt + prompt
        try:
            logger.info("Checking hallucination for answer")
            return self._generate_response(messages, max_tokens=150, temperature=0).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)

class OpenAi:
    def __init__(self, api_key, model):
        """
        Initializes the GPT-3 model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT-3 model version.
        """
        from openai import OpenAI

        try:
            logger.info(f"Initializing OpenAi with model: {model}")
            self.model = model
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages, max_tokens=250, temperature=0):
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_prompt(self, context, question):
        """
        Formats the prompt for generating a response.

        Args:
            context (str): The context for the prompt.
            question (str): The question for the prompt.

        Returns:
            str: The formatted prompt.
        """
        return f"Given Context: {context} Give the best full answer amongst the option to question {question}"

    def answer_question(self, context, question, max_tokens=350):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 350.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Answering question")
            prompt = self._format_prompt(context, question)
            messages = [
                {"role": "system", "content": "You are Question Answering Portal"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages, max_tokens=max_tokens, temperature=0)
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
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages, max_tokens=150, temperature=0)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                grade = self._generate_response(messages, max_tokens=150, temperature=0).lower()
                if grade == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logger.info("Not relevant doc")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            logger.info("Checking hallucination for answer")
            return self._generate_response(messages, max_tokens=150, temperature=0).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)
