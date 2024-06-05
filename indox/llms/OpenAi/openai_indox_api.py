import requests


class IndoxApiOpenAiQa:
    def __init__(self, api_key, prompt_template=None):
        """
        Initializes the IndoxApiOpenAiQa with the specified API key and an optional prompt template.

        Args:
            api_key (str): The API key for Indox API.
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        self.api_key = api_key
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, prompt):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": 150,
            "messages": prompt,
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
        prompt = self.prompt_template.format(context=context, question=question)
        return self._send_request(prompt)

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
            prompt = [
                {
                    "content": "you are a helpful assistant.",
                    "role": "system"
                },
                {
                    "content": "You are a helpful assistant. Give a detailed summary of the documentation "
                               "provided.\n\nDocumentation:\n" + documentation,
                    "role": "user"
                }
            ]
            return self._send_request(prompt)
        except Exception as e:
            print(e)
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
            # system_prompt = f"""
            # Answer the following questions and obey the following commands as best you can.
            #
            # You have access to the following tools:
            #
            # {tool_description}
            #
            # You will receive a message from the human, then you should start a loop and do one of two things
            #
            # Option 1: You use a tool to answer the question.
            # For this, you should use the following format:
            # Thought: you should always think about what to do
            # Action: the action to take, should be one of [{tool_names}]
            # Action Input: "the input to the action, to be sent to the tool"
            #
            # After this, the human will respond with an observation, and you will continue.
            #
            # Option 2: You respond to the human.
            # For this, you should use the following format:
            # Action: Response To Human
            # Action Input: "your response to the human, summarizing what you did and what you learned"
            #
            # Begin!
            # """
            # prompt = [
            #     {
            #         "content": system_prompt,
            #         "role": "system"
            #     },
            #     {
            #         "content": f"Context: {context}\nQuestion: {question}\n",
            #         "role": "user"
            #     }
            # ]
            prompt = question
            return self._send_request(prompt)
        except Exception as e:
            print(e)
            return str(e)
