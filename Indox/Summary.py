from openai import OpenAI
from .utils import read_config
from transformers import pipeline
import logging


# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
# )


def summarize(context):
    """
    Summarizes the given context using either OpenAI's GPT-3.5 model or a Hugging Face model based on the configuration.

    Args:
        context (str): The input context to be summarized.

    Returns:
        tuple: A tuple containing the summarized text, input tokens used for the summary, and output tokens generated.
        (if you use openai). there is no return other than summary itself if you use huggingface model.
    """
    config = read_config()
    try:
        if config["summary_model"]["model_name"] == "gpt-3.5-turbo-0125":
            client = OpenAI()

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": config["prompts"]["summary_model"]["content"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}",
                    },
                ],
                max_tokens=config["summary_model"]["max_tokens"],
                model="gpt-3.5-turbo-0125",
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
            return response.choices[0].message.content.replace("\n", " "), input_tokens, output_tokens
        else:
            hf_model = SummarizationModelHuggingFace(
                config["summary_model"]["model_name"],
                max_len=config["summary_model"]["max_tokens"],
                min_len=config["summary_model"]["min_len"],
            )
            return hf_model.summarize(context), 0, None

    except Exception as e:
        print(e)
        return e


class SummarizationModelHuggingFace:
    def __init__(self, model_name, max_len=100, min_len=30):
        self.model_name = model_name
        self.max_len = max_len
        self.min_len = min_len
        logging.info(f"loading the custom model: {model_name}")
        self.summarizer = self.load_model(
            model_name,
        )
        logging.info(f"Model loaded successfuly. ")

    def load_model(self, model_name):
        try:
            summarizer = pipeline("summarization", model=model_name)
            return summarizer
        except:
            raise NameError(f"Can't load the model: {model_name}")

    def summarize(self, context):
        if self.summarizer is None:
            raise RuntimeError(f"No model is provided")

        return self.summarizer(
            context, max_length=self.max_len, min_length=self.min_len, do_sample=False
        )[0]["summary_text"]
