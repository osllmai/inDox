# class ContextualRelevancyTemplate:
#     @staticmethod
#     def generate_reason(query: str, irrelevancies: str, score: float) -> str:
#         return f"""Based on the given input, reasons for why the retrieval context is irrelevant to the input,
#          and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
# In your reason, you should quote data provided in the reasons for irrelevancy to support your point.

# **
# IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
# Example JSON:
# {{
#     "reason": "The score is <contextual_relevancy_score> because <your_reason>."
# }}

# If the score is 1, keep it short and say something positive with an upbeat encouraging tone
# (but don't overdo it otherwise it gets annoying).
# **

# Contextual Relevancy Score:
# {score}

# Input:
# {query}

# Reasons for why the retrieval context is irrelevant to the input:
# {irrelevancies}

# JSON:
# """

#     @staticmethod
#     def generate_verdict(query: str, context: str) -> str:
#         return f"""Based on the input and context, please generate a JSON object to indicate whether the context
#         is relevant to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'reason'.
#     The 'verdict' key should STRICTLY be either 'yes', 'partial', or 'no', indicating whether the context is relevant, partially relevant, or irrelevant to the input.
#     Provide a 'reason' explaining your choice. If the verdict is 'no', you MUST quote the irrelevant parts of the context
#     to back up your reason. If the verdict is 'yes' or 'partial', explain why the context is relevant or partially relevant.

#     **
#     IMPORTANT: Please make sure to only return in JSON format.
#     Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in
#     1968. There was a cat."
#     Example Input: "What were some of Einstein's achievements?"

#     Example:
#     {{
#         "verdict": "no",
#         "reason": "Although the context contains information about Einstein winning the Nobel Prize, it irrelevantly includes
#         'There was a cat' when it has nothing to do with Einstein's achievements."
#     }},
#     {{
#         "verdict": "partial",
#         "reason": "The context mentions Einstein's Nobel Prize for his discovery of the photoelectric effect, which is relevant, but includes incorrect information about the year and irrelevant mention of a cat."
#     }},
#     {{
#         "verdict": "yes",
#         "reason": "The context is relevant because it mentions Einstein's Nobel Prize for his discovery of the photoelectric effect, which is a significant achievement."
#     }}

#     Input:
#     {query}

#     Context:
#     {context}

#     JSON:
#     """

from typing import List


class ContextualRelevancyTemplate:
    @staticmethod
    def generate_verdict(query: str, context: str) -> str:
        return f"""Based on the input and context, please generate a JSON object to indicate whether the context 
        is relevant to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'reason'.
        The 'verdict' key should STRICTLY be either 'yes', 'partial', or 'no', indicating whether the context is relevant, 
        partially relevant, or irrelevant to the input. 
        Provide a 'reason' explaining your choice. If the verdict is 'no', you MUST quote the irrelevant parts of the context 
        to back up your reason. If the verdict is 'yes' or 'partial', explain why the context is relevant or partially relevant.

        **
        IMPORTANT: Please make sure to only return in JSON format.
        Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 
        1968. There was a cat."
        Example Input: "What were some of Einstein's achievements?"

        Example:
        {{
            "verdict": "no",
            "reason": "Although the context contains information about Einstein winning the Nobel Prize, it irrelevantly includes 
            'There was a cat' when it has nothing to do with Einstein's achievements."
        }}

        Input:
        {query}

        Context:
        {context}

        JSON:
        """

    @staticmethod
    def generate_reason(query: str, irrelevancies: List[str], score: float) -> str:
        return f"""Based on the given input, reasons for why the retrieval context is irrelevant to the input,
         and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
         In your reason, you should quote data provided in the reasons for irrelevancy to support your point.

        ** 
        IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
        Example JSON:
        {{
            "reason": "The score is <contextual_relevancy_score> because <your_reason>."
        }}

        If the score is 1, keep it short and say something positive with an upbeat encouraging tone 
        (but don't overdo it otherwise it gets annoying).
        **

        Contextual Relevancy Score:
        {score}

        Input:
        {query}

        Reasons for why the retrieval context is irrelevant to the input:
        {irrelevancies}

        JSON:
        """
