import logging
import requests

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class IndoxApiOpenAi:
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


    # def _send_request(self, system_prompt, user_prompt, role: str = "system", history: list = None):
    #     url = 'http://5.78.55.161/api/chat_completion/generate/'
    #     headers = {
    #         'accept': '*/*',
    #         "Authorization": f"Bearer {self.api_key}",
    #         'Content-Type': 'application/json',
    #     }
    #
    #     data = {
    #         "frequency_penalty": 0,
    #         "max_tokens": 150,
    #         "messages": [
    #             {
    #                 "content": system_prompt,
    #                 "role": role
    #             },
    #             {
    #                 "content": user_prompt,
    #                 "role": "user"
    #             }
    #         ],
    #         "model": "gpt-3.5-turbo-0125",
    #         "presence_penalty": 0,
    #         "stream": True,
    #         "temperature": 0.3,
    #         "top_p": 1
    #     }
    #     if history is not None and len(history) > 2:
    #         assert isinstance(history, list)
    #         data['messages'] = history + data['messages']
    #
    #     response = requests.post(url, headers=headers, json=data)
    #     if response.status_code == 200:
    #         answer_data = response.json()
    #         generated_text = answer_data.get("text_message", "")
    #         return generated_text
    #     else:
    #         raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")


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
            prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(context, question)
        except Exception as e:
            print(e)
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
            user_prompt = "Give a detailed summary of the documentation provided.\n\nDocumentation:\n" + documentation
            return self._send_request(system_prompt, user_prompt)
        except Exception as e:
            print(e)
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
                    print("Relevant doc")
                    filtered_docs.append(context[i])
                elif response.lower() == "no":
                    print("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
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
            logging.error("Error generating agent answer: %s", e)
            return str(e)

    def assistant(self, system_prompt, user_prompt, role, history):
        try:
            response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt, role=role, history=history)
            return response
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
            return str(e)
