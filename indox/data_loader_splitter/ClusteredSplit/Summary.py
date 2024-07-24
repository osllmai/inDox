# import os
# from typing import List, Optional, Tuple
# import requests
# from openai import OpenAI
# from dotenv import load_dotenv
#
# load_dotenv()


#
#
#
#
#
# def summarize(context):
#     """
#     Summarizes the given context using either OpenAI's GPT-3.5 model or a Hugging Face model based on the configuration.
#
#     Args:
#         context (str): The input context to be summarized.
#
#     Returns:
#         tuple: A tuple containing the summarized text, input tokens used for the summary, and output tokens generated.
#         (if you use openai). there is no return other than summary itself if you use huggingface model.
#     """
#     config = read_config()
#     try:
#         if config["summary_model"]["model_name"] == "gpt-3.5-turbo-0125":
#             client = OpenAI()
#
#             response = client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": config["prompts"]["summary_model"]["content"],
#                     },
#                     {
#                         "role": "user",
#                         "content": f"{context}",
#                     },
#                 ],
#                 max_tokens=config["summary_model"]["max_tokens"],
#                 model="gpt-3.5-turbo-0125",
#             )
#
#             return response.choices[0].message.content.replace("\n", " ")
#         else:
#             hf_model = SummarizationModelHuggingFace(
#                 config["summary_model"]["model_name"],
#                 max_len=config["summary_model"]["max_tokens"],
#                 min_len=config["summary_model"]["min_len"],
#             )
#             return hf_model.summarize(context), 0, None
#
#     except Exception as e:
#         print(e)
#         return e
#
#
# class SummarizationModelHuggingFace:
#     def __init__(self, model_name, max_len=100, min_len=30):
#         self.model_name = model_name
#         self.max_len = max_len
#         self.min_len = min_len
#         logging.info(f"loading the custom model: {model_name}")
#         self.summarizer = self.load_model(
#             model_name,
#         )
#         logging.info(f"Model loaded successfuly. ")
#
#     def load_model(self, model_name):
#         try:
#             summarizer = pipeline("summarization", model=model_name)
#             return summarizer
#         except:
#             raise NameError(f"Can't load the model: {model_name}")
#
#     def summarize(self, context):
#         if self.summarizer is None:
#             raise RuntimeError(f"No model is provided")
#
#         return self.summarizer(
#             context, max_length=self.max_len, min_length=self.min_len, do_sample=False
#         )[0]["summary_text"]


# prompt_content = "You are a helpful assistant. Give a detailed summary of the documentation provided"
# RANDOM_SEED = 42  # Fixed seed for reproducibility


# RANDOM_SEED = 42


# Base class for summary models
class SummaryModel:
    def __init__(self, model_name: str, max_len: int, min_len: int):
        self.model_name = model_name
        self.max_len = max_len
        self.min_len = min_len

    def summarize(self, context: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")


# OpenAI summary model class
class OpenAISummaryModel(SummaryModel):
    def __init__(self, model_name: str, max_len: int, min_len: int, prompt_content: str):
        super().__init__(model_name, max_len, min_len)
        self.prompt_content = prompt_content
        self.client = OpenAI()

    def summarize(self, context: str):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_content,
                },
                {
                    "role": "user",
                    "content": f"{context}",
                },
            ],
            max_tokens=self.max_len,
            model=self.model_name,
        )

        return response.choices[0].message.content.strip()


# Hugging Face summary model class
class HuggingFaceSummaryModel(SummaryModel):
    def __init__(self, model_name: str, max_len: int, min_len: int):
        super().__init__(model_name, max_len, min_len)
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.max_len = max_len
        self.min_len = min_len

    def summarize(self, context: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": context,
            "parameters": {
                "max_length": self.max_len,
                "min_length": self.min_len,
                "do_sample": False,
            }
        }
        for _ in range(3):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()[0]["summary_text"]
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
        # response = requests.post(self.api_url, headers=headers, json=payload)
        # response.raise_for_status()
        # return response.json()[0]["summary_text"]


# Main function to summarize context
def summarize(context: str, use_openai: bool, max_len: int, min_len: int):
    """
    Summarizes the given context using either OpenAI's GPT-3.5 model or a Hugging Face model based on the parameter.

    Args:
        context (str): The input context to be summarized.
        use_openai (bool): Flag to indicate whether to use OpenAI model (True) or Hugging Face model (False).
        max_len (int): The maximum length of the summary text.
        min_len (int): The minimum length of the summary text.

    Returns:
        tuple: A tuple containing the summarized text, input tokens used for the summary, and output tokens generated
               if using OpenAI, otherwise just the summary text.
    """
    # Parameters for the summarization models
    openai_model_name = "gpt-3.5-turbo-0125"
    huggingface_model_name = "facebook/bart-large-cnn"
    max_len = max_len
    min_len = min_len
    prompt_content = "You are a helpful assistant. Give a detailed summary of the documentation provided"

    try:
        if use_openai:
            summary_model = OpenAISummaryModel(
                model_name=openai_model_name,
                max_len=max_len,
                min_len=min_len,
                prompt_content=prompt_content
            )
            return summary_model.summarize(context)
        else:
            summary_model = HuggingFaceSummaryModel(
                model_name=huggingface_model_name,
                max_len=max_len,
                min_len=min_len
            )
            return summary_model.summarize(context)

    except Exception as e:
        print(e)
        return e
