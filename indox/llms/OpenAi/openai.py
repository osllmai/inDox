import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import os

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class OpenAi:
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
        
            Give a binary score ''yes'' or ''no'' score to indicate whether the document is
            relevant to the question.
        
            Provide the score with no preamble or explanation.
            """
            for i in range(len(context)):
                prompt = f"""
                    Here is the retrieved document:
                    {context[i]}
                    Here is the user question: 
                    {question}"""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=150,
                )
                grade = response.choices[0].message.content.strip()
                if grade.lower() == "yes":
                    print("Relevant doc")
                    filtered_docs.append(context[i])
                elif grade.lower() == "no":
                    print("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
            )
            hallucination_answer = response.choices[0].message.content.strip()
            logging.info("Agent answer generated successfully")
            return hallucination_answer
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
            return str(e)
