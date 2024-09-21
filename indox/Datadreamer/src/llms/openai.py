import gc
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, cast
import requests
import openai
import tiktoken
from datasets.fingerprint import Hasher
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)
from tiktoken import Encoding

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_temperature_and_top_p,
)


@lru_cache(maxsize=None)
def _normalize_model_name(model_name: str) -> str:
    if ":" in model_name:  # pragma: no cover
        # Handles extracting the model name from a fine-tune
        # model name like "ft:babbage-002:org:datadreamer:xxxxxxx"
        model_name = model_name.split(":")[1]
    return model_name


@lru_cache(maxsize=None)
def _is_gpt_3(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(
        gpt3_name in model_name for gpt3_name in ["davinci", "ada", "curie", "gpt-3-"]
    )


@lru_cache(maxsize=None)
def _is_gpt_3_5(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt35_name in model_name for gpt35_name in ["gpt-3.5-", "gpt-35-"])


@lru_cache(maxsize=None)
def _is_gpt_3_5_legacy(model_name: str):
    model_name = _normalize_model_name(model_name)
    return _is_gpt_3_5(model_name) and (
        "-0613" in model_name
        or (_is_instruction_tuned(model_name) and not _is_chat_model(model_name))
    )


@lru_cache(maxsize=None)
def _is_gpt_4(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        model_name == "gpt-4"
        or any(gpt4_name in model_name for gpt4_name in ["gpt-4-"])
        or _is_gpt_4o(model_name)
    )


@lru_cache(maxsize=None)
def _is_gpt_4o(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt4_name in model_name for gpt4_name in ["gpt-4o"])


@lru_cache(maxsize=None)
def _is_gpt_mini(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt_mini_name in model_name for gpt_mini_name in ["-mini"])


@lru_cache(maxsize=None)
def _is_128k_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return _is_gpt_4(model_name) and (
        _is_gpt_4o(model_name) or "-preview" in model_name or "2024-04-09" in model_name
    )


@lru_cache(maxsize=None)
def _is_chat_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_gpt_3_5(model_name) or _is_gpt_4(model_name)
    ) and not model_name.endswith("-instruct")


@lru_cache(maxsize=None)
def _is_instruction_tuned(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_chat_model(model_name)
        or model_name.startswith("text-")
        or model_name.endswith("-instruct")
    )


class OpenAIException(Exception):
    pass


class OpenAI(LLM):
    def __init__(self, api_key, model_name, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.chat_prompt_template = kwargs.get("chat_prompt_template", None)
        self.system_prompt = kwargs.get("system_prompt", None)

    @lru_cache(maxsize=None)
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the input text."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to a known encoding, like gpt-4 or another supported model
            encoding = tiktoken.get_encoding("cl100k_base")  # or another supported encoding
        return len(encoding.encode(text))

    @lru_cache(maxsize=None)
    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        """Returns the maximum context length allowed by the model."""
        # Convert the tuple back to a list for processing if necessary
        return 7000 - max_new_tokens

    def run(self, prompts: Iterable[str], max_new_tokens: int = 0, **kwargs) -> Generator[str, None, None]:
        """
        Runs the model with multiple prompts and returns a generator of responses.

        Args:
            prompts (Iterable[str]): A list or generator of prompts to process.
            max_new_tokens (int): The maximum number of new tokens that can be generated.
            **kwargs: Additional arguments passed, like temperature, top_p, etc.

        Yields:
            str: Generated response for each prompt.
        """
        # Convert prompts to a tuple to avoid TypeError
        prompts_tuple = tuple(prompts)
        print(prompts_tuple)

        for prompt in prompts_tuple:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            yield self.generate_response(system_prompt, prompt, **kwargs)

    def _send_request(self, system_prompt, user_prompt, **kwargs):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": kwargs.get("max_new_tokens", 150),
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
            "model": f"{self.model_name}",
            "presence_penalty": 0,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 1)
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")

    def generate_response(self, system_prompt, user_prompt, **kwargs):
        """Method to simplify calling the _send_request."""
        return self._send_request(system_prompt, user_prompt, **kwargs)

    @cached_property
    def display_name(self) -> str:
        return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        to_hash: list[Any] = []
        names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached client and tokenizer
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached client or tokenizer before serializing
        state.pop("retry_wrapper", None)
        state.pop("client", None)
        state.pop("tokenizer", None)

        # Check if executor_pools exists before attempting to clear it
        if "executor_pools" in state:
            state["executor_pools"].clear()

        return state
