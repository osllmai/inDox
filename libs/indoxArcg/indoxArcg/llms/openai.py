from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


def format_prompt(context, question):
    """
    Formats the prompt for generating a response.

    Args:
        context (str): The context for the prompt.
        question (str): The question for the prompt.

    Returns:
        str: The formatted prompt.
    """
    return f"Given Context: {context} Give the best full answer amongst the option to question {question}"


class OpenAi:
    def __init__(self, api_key, model, base_url=None):
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
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
            logger.info("OpenAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAi: {e}")
            raise

    def _generate_response(
        self,
        messages,
        max_tokens,
        temperature,
        frequency_penalty,
        presence_penalty,
        top_p,
        stream,
    ):
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model.
            max_tokens (int): The maximum number of tokens in the generated response.
            temperature (float): The sampling temperature.
            frequency_penalty (float): The frequency penalty.
            presence_penalty (float): The presence penalty.
            top_p (float): The top_p parameter for nucleus sampling.
            stream: Indicates if the response should be streamed.

        Returns:
            str: The generated response.
        """
        try:
            # logger.info("Generating response")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                stream=stream,
            )

            if stream:
                # If streaming, accumulate the response content
                result = ""
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content is not None:
                        result += content
                result = result.strip()
            else:
                # For non-streaming response
                result = response.choices[0].message.content.strip()

            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def chat(
        self,
        prompt,
        system_prompt="You are a helpful assistant",
        max_tokens=None,
        temperature=0.2,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        stream=None,
    ):
        """
        Public method to interact with the model using chat messages.

        Args:
            prompt (str): The prompt to generate a response for.
            system_prompt (str): The system prompt.
            max_tokens (int, optional): The maximum number of tokens in the generated response.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The nucleus sampling parameter.
            stream: Whether to stream the response.

        Returns:
            str: The generated response.
        """
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            stream=stream,
        )

    def answer_question(
        self,
        context,
        question,
        max_tokens=350,
        temperature=0.3,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        stream=False,
    ):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated response.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The top_p parameter for nucleus sampling.
            stream (bool, optional): Whether to stream the response.

        Returns:
            str: The generated answer.
        """
        try:
            prompt = format_prompt(context, question)
            messages = [
                {"role": "system", "content": "You are Question Answering Portal"},
                {"role": "user", "content": prompt},
            ]
            response = self._generate_response(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                stream=stream,
            )
            return response
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return str(e)

    def get_summary(
        self,
        documentation,
        max_tokens=350,
        temperature=0.3,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
    ):
        """
        Generates a detailed summary of the provided documentation.

        Args:
            documentation (str): The documentation to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated response.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The top_p parameter for nucleus sampling.

        Returns:
            str: The generated summary.
        """
        try:
            logger.info("Generating summary for documentation")
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            messages = [
                {"role": "developer", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                stream=False,
            )
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return str(e)

    def grade_docs(
        self,
        context,
        question,
        max_tokens=150,
        temperature=0.2,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
    ):
        """
        Grades documents for relevance to a question using the LLM.

        Args:
            context (list): The context in which the question is asked.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated response.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The top_p parameter for nucleus sampling.

        Returns:
            list: Filtered relevant documents.
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
            prompt = f"Here is the retrieved document:\n{doc.strip().replace('\\n', ' ')}\nHere is the user question:\n{question}"
            messages = [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                grade = (
                    self._generate_response(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        top_p=top_p,
                        stream=False,
                    )
                    .strip()
                    .lower()
                )

                if grade == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logger.info("Not relevant doc")
                else:
                    logger.warning(f"Unexpected grade response: {grade}")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
                logger.error("Skipping this document due to an error.")

        return filtered_docs

    def check_hallucination(
        self,
        context,
        answer,
        max_tokens=150,
        temperature=0.2,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
    ):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The context for checking.
            answer (str): The answer to check.
            max_tokens (int, optional): The maximum number of tokens in the generated response.
            temperature (float, optional): The temperature of the generated response.
            frequency_penalty (float, optional): The frequency penalty.
            presence_penalty (float, optional): The presence penalty.
            top_p (float, optional): The top_p parameter for nucleus sampling.

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
            return self._generate_response(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                stream=False,
            ).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)
