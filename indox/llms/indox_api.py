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
            "max_tokens" : max_tokens,
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

    def answer_question(self, context, question, max_tokens=350, temperature=0.3, stream=True,
                        model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
        """
        Answer a question based on the given context using the Indox API.

        Returns:
            str: The generated answer.
        """
        try:
            # prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(context, question, max_tokens=max_tokens,
                                                 temperature=temperature, stream=stream,
                                                 model=model, presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty,
                                                 top_p=top_p)
        except Exception as e:
            logger.error(e)
            return str(e)

    def get_summary(self, documentation, max_tokens=350, temperature=0.3, stream=True,
                    model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
        """
        Generates a detailed summary of the provided documentation.


        Returns:
            str: The generated summary.
        """
        try:
            system_prompt = "You are a helpful assistant."
            user_prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation}"
            return self._send_request(system_prompt=system_prompt, user_prompt=user_prompt,
                                      max_tokens=max_tokens,
                                      temperature=temperature, stream=stream,
                                      model=model, presence_penalty=presence_penalty,
                                      frequency_penalty=frequency_penalty,
                                      top_p=top_p)
        except Exception as e:
            logger.error(e)
            return str(e)

    def grade_docs(self, context, question, max_tokens=350, temperature=0.3, stream=True,
                   model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
        """
        Answers a question using an agent-based approach with access to tools.

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
                response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt,
                                              max_tokens=max_tokens,
                                              temperature=temperature, stream=stream,
                                              model=model, presence_penalty=presence_penalty,
                                              frequency_penalty=frequency_penalty,
                                              top_p=top_p)
                if response.lower() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif response.lower() == "no":
                    logger.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def check_hallucination(self, context, answer, max_tokens=350, temperature=0.3, stream=True,
                            model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
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
            response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens,
                                          temperature=temperature, stream=stream,
                                          model=model, presence_penalty=presence_penalty,
                                          frequency_penalty=frequency_penalty,
                                          top_p=top_p)
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    # def _assistant(self, system_prompt, user_prompt, role, history):
    #     try:
    #         response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt, role=role,
    #                                       history=history)
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error generating agent answer: {e}")
    #         return str(e)

    def chat(self, prompt, system_prompt="You are a helpful assistant", max_tokens=16384, temperature=0.3, stream=True,
             model="gpt-4o-mini", presence_penalty=0, frequency_penalty=0, top_p=1):
        return self._send_request(system_prompt=system_prompt, user_prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, stream=stream,
                                  model=model, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                                  top_p=top_p)
