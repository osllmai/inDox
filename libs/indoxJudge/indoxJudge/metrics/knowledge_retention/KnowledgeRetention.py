# from typing import List, Dict, Union
# from pydantic import BaseModel, Field

# import json

# from .template import KnowledgeRetentionTemplate

# class Knowledge(BaseModel):
#     """
#     Model representing a dictionary of knowledge data, which can include strings and lists of strings.
#     """
#     data: Dict[str, Union[str, List[str]]]

# class KnowledgeRetentionVerdict(BaseModel):
#     """
#     Model representing a verdict on the retention of knowledge in the LLM response,
#     including the index of the message, the verdict itself, and the reasoning behind it.
#     """
#     index: int
#     verdict: str
#     reason: str = Field(default=None)

# class KnowledgeRetention:
#     """
#     Class for evaluating the retention of knowledge in language model outputs by analyzing the continuity of knowledge
#     across multiple messages, generating verdicts, and calculating retention scores.
#     """
#     def __init__(self, messages: List[Dict[str, str]], threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
#         """
#         Initializes the KnowledgeRetention class with the messages, threshold, and evaluation settings.

#         Parameters:
#         messages (List[Dict[str, str]]): A list of messages containing queries and LLM responses.
#         threshold (float): The threshold for determining successful knowledge retention. Defaults to 0.5.
#         include_reason (bool): Whether to include reasoning for the knowledge retention verdicts. Defaults to True.
#         strict_mode (bool): Whether to use strict mode, which forces a score of 0 if retention is below the threshold. Defaults to False.
#         """
#         self.model = None
#         self.messages = messages
#         self.threshold = 1 if strict_mode else threshold
#         self.include_reason = include_reason
#         self.strict_mode = strict_mode
#         self.knowledges = []
#         self.verdicts = []
#         self.reason = None
#         self.score = None
#         self.success = None

#     def set_model(self, model):
#         """
#         Sets the language model to be used for evaluation.

#         Parameters:
#         model: The language model to use.
#         """
#         self.model = model

#     def measure(self) -> float:
#         """
#         Measures the level of knowledge retention in the LLM responses by generating knowledges and verdicts,
#         then calculating the retention score.

#         Returns:
#         float: The calculated knowledge retention score.
#         """
#         self.knowledges = self._generate_knowledges()
#         self.verdicts = self._generate_verdicts()
#         knowledge_retention_score = self._calculate_score()
#         self.reason = self._generate_reason(knowledge_retention_score)
#         self.success = knowledge_retention_score >= self.threshold
#         self.score = knowledge_retention_score
#         return self.score

#     def _generate_reason(self, score: float) -> str:
#         """
#         Generates the reasoning behind the knowledge retention score if include_reason is set to True.

#         Parameters:
#         score (float): The calculated knowledge retention score.

#         Returns:
#         str: The reasoning behind the knowledge retention score.
#         """
#         if not self.include_reason:
#             return None

#         attritions = [verdict.reason for verdict in self.verdicts if verdict.verdict.strip().lower() == "yes"]

#         prompt = KnowledgeRetentionTemplate.generate_reason(
#             attritions=attritions,
#             score=format(score, ".2f"),
#         )

#         response = self._call_language_model(prompt)
#         data = json.loads(response)
#         return data["reason"]

#     def _calculate_score(self) -> float:
#         """
#         Calculates the knowledge retention score based on the number of retained knowledges.

#         Returns:
#         float: The calculated knowledge retention score.
#         """
#         number_of_verdicts = len(self.verdicts)
#         if number_of_verdicts == 0:
#             return 0

#         retention_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no")

#         score = retention_count / number_of_verdicts

#         return 0 if self.strict_mode and score < self.threshold else score

#     def _generate_verdicts(self) -> List[KnowledgeRetentionVerdict]:
#         """
#         Generates a list of verdicts on the retention of knowledge for each message.

#         Returns:
#         List[KnowledgeRetentionVerdict]: A list of KnowledgeRetentionVerdict instances.
#         """
#         verdicts = []
#         for index, message in enumerate(self.messages):
#             previous_knowledge = self.knowledges[index].data

#             prompt = KnowledgeRetentionTemplate.generate_verdict(
#                 llm_message=message["llm_response"],
#                 previous_knowledge=previous_knowledge,
#             )
#             response = self._call_language_model(prompt)
#             data = json.loads(response)
#             verdict = KnowledgeRetentionVerdict(index=index, **data)
#             verdicts.append(verdict)

#         return verdicts

#     def _generate_knowledges(self) -> List[Knowledge]:
#         """
#         Generates a list of knowledge data for each message.

#         Returns:
#         List[Knowledge]: A list of Knowledge instances.
#         """
#         knowledges = []
#         for index, message in enumerate(self.messages):
#             previous_knowledge = knowledges[-1].data if knowledges else {}
#             llm_message = self.messages[index - 1]["llm_response"] if index > 0 else ""

#             prompt = KnowledgeRetentionTemplate.extract_data(
#                 llm_message=llm_message,
#                 user_message=message["query"],
#                 previous_knowledge=previous_knowledge,
#             )

#             response = self._call_language_model(prompt)
#             data = json.loads(response)
#             knowledge = Knowledge(data=data)
#             knowledges.append(knowledge)

#         return knowledges

#     def _call_language_model(self, prompt: str) -> str:
#         """
#         Calls the language model with the given prompt and returns the response.

#         Parameters:
#         prompt (str): The prompt to provide to the language model.

#         Returns:
#         str: The response from the language model.
#         """
#         response = self.model.generate_evaluation_response(prompt=prompt)
#         return response


from typing import List, Dict, Union
from pydantic import BaseModel, Field
import json

from .template import KnowledgeRetentionTemplate
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


class Knowledge(BaseModel):
    """
    Model representing a dictionary of knowledge data, which can include strings and lists of strings.
    """

    data: Dict[str, Union[str, List[str]]]


class KnowledgeRetentionVerdict(BaseModel):
    """
    Model representing a verdict on the retention of knowledge in the LLM response,
    including the index of the message, the verdict itself, and the reasoning behind it.
    """

    index: int
    verdict: str
    reason: str = Field(default=None)


class KnowledgeRetention:
    """
    Class for evaluating the retention of knowledge in language model outputs by analyzing the continuity of knowledge
    across multiple messages, generating verdicts, and calculating retention scores.
    """

    def __init__(
        self,
        messages: List[Dict[str, str]],
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initializes the KnowledgeRetention class with the messages, threshold, and evaluation settings.

        Parameters:
        messages (List[Dict[str, str]]): A list of messages containing queries and LLM responses.
        threshold (float): The threshold for determining successful knowledge retention. Defaults to 0.5.
        include_reason (bool): Whether to include reasoning for the knowledge retention verdicts. Defaults to True.
        strict_mode (bool): Whether to use strict mode, which forces a score of 0 if retention is below the threshold. Defaults to False.
        """
        self.model = None
        self.messages = messages
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.knowledges = []
        self.verdicts = []
        self.reason = None
        self.score = None
        self.success = None
        self.total_output_tokens = 0
        self.total_input_tokens = 0

    def set_model(self, model):
        """
        Sets the language model to be used for evaluation.

        Parameters:
        model: The language model to use.
        """
        self.model = model

    def _clean_json_response(self, response: str) -> str:
        """
        Cleans the JSON response from the language model by removing markdown code blocks if present.

        :param response: Raw response from the language model
        :return: Cleaned JSON string
        """
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def measure(self) -> float:
        """
        Measures the level of knowledge retention in the LLM responses by generating knowledges and verdicts,
        then calculating the retention score.

        Returns:
        float: The calculated knowledge retention score.
        """
        self.knowledges = self._generate_knowledges()
        self.verdicts = self._generate_verdicts()
        knowledge_retention_score = self._calculate_score()
        self.reason = self._generate_reason(knowledge_retention_score)
        self.success = knowledge_retention_score >= self.threshold
        self.score = knowledge_retention_score
        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | Total Output: {self.total_output_tokens} | Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return self.score

    def _generate_reason(self, score: float) -> str:
        """
        Generates the reasoning behind the knowledge retention score if include_reason is set to True.

        Parameters:
        score (float): The calculated knowledge retention score.

        Returns:
        str: The reasoning behind the knowledge retention score.
        """
        if not self.include_reason:
            return None

        attritions = [
            verdict.reason
            for verdict in self.verdicts
            if verdict.verdict.strip().lower() == "yes"
        ]

        prompt = KnowledgeRetentionTemplate.generate_reason(
            attritions=attritions,
            score=format(score, ".2f"),
        )

        response = self._call_language_model(prompt)
        cleaned_response = self._clean_json_response(response)
        data = json.loads(cleaned_response)
        return data["reason"]

    def _calculate_score(self) -> float:
        """
        Calculates the knowledge retention score based on the number of retained knowledges.

        Returns:
        float: The calculated knowledge retention score.
        """
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        retention_count = sum(
            1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no"
        )

        score = retention_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_verdicts(self) -> List[KnowledgeRetentionVerdict]:
        """
        Generates a list of verdicts on the retention of knowledge for each message.

        Returns:
        List[KnowledgeRetentionVerdict]: A list of KnowledgeRetentionVerdict instances.
        """
        verdicts = []
        for index, message in enumerate(self.messages):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=message["llm_response"],
                previous_knowledge=previous_knowledge,
            )
            response = self._call_language_model(prompt)
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)
            verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    def _generate_knowledges(self) -> List[Knowledge]:
        """
        Generates a list of knowledge data for each message.

        Returns:
        List[Knowledge]: A list of Knowledge instances.
        """
        knowledges = []
        for index, message in enumerate(self.messages):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_message = self.messages[index - 1]["llm_response"] if index > 0 else ""

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_message,
                user_message=message["query"],
                previous_knowledge=previous_knowledge,
            )

            response = self._call_language_model(prompt)
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)
            knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

    def _clean_json_response(self, response: str) -> str:
        """
        Cleans the JSON response from the language model by removing markdown code blocks if present.

        :param response: Raw response from the language model
        :return: Cleaned JSON string
        """
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(prompt=prompt)
        self.total_input_tokens += input_token_count

        if not response:
            raise ValueError("Received an empty response from the model.")

        clean_response = self._clean_json_response(response=response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response
