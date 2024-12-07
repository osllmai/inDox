from abc import ABC, abstractmethod
import requests
from loguru import logger
import sys
import httpx

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class BaseLLM(ABC):
    """
    Base class for LLM (Large Language Model) providers.

    This class defines the common interface that all LLM providers (e.g., OpenAI, Anthropic) must implement.

    Methods:
        generate(prompt: str) -> str: 
            Abstract method to generate a response from the model based on a given prompt.
    
    Example:
        # This is an abstract base class; no direct instantiation.
        pass
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class AsyncOpenAi(BaseLLM):
    """
    Asynchronous OpenAI provider with enhanced error handling.

    This class interacts with OpenAI's API asynchronously to generate text completions.

    Attributes:
        api_key (str): The API key for authenticating with OpenAI.
        model (str): The model to use (default: "gpt-4").
        temperature (float): The temperature to control randomness (default: 0.0).
        max_tokens (int): The maximum number of tokens to generate (default: 2000).

    Methods:
        generate(prompt: str) -> str:
            Asynchronously generates a response from OpenAI based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = AsyncOpenAi(api_key="your-api-key")
        # result = await ai_provider.generate("Tell me a joke.")
    """


    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        base_url: str = None,
    ):
        from openai import AsyncOpenAI

        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OpenAi(BaseLLM):
    """
    Synchronous OpenAI provider with enhanced error handling.

    This class interacts with OpenAI's API synchronously to generate text completions.

    Attributes:
        api_key (str): The API key for authenticating with OpenAI.
        model (str): The model to use (default: "gpt-4").
        temperature (float): The temperature to control randomness (default: 0.0).
        max_tokens (int): The maximum number of tokens to generate (default: 2000).

    Methods:
        generate(prompt: str) -> str:
            Synchronously generates a response from OpenAI based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = OpenAi(api_key="your-api-key")
        # result = ai_provider.generate("Tell me a joke.")
    """


    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        base_url: str = None,
    ):
        from openai import OpenAI

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AsyncAnthropic(BaseLLM):
    """
    Asynchronous Anthropic Claude provider with enhanced error handling.

    This class interacts with Anthropic's Claude API asynchronously to generate text completions.

    Attributes:
        api_key (str): The API key for authenticating with Anthropic.
        model (str): The model to use (default: "claude-3-opus-20240229").
        temperature (float): The temperature to control randomness (default: 0.0).
        max_tokens (int): The maximum number of tokens to generate (default: 2000).

    Methods:
        generate(prompt: str) -> str:
            Asynchronously generates a response from Anthropic Claude based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = AsyncAnthropic(api_key="your-api-key")
        # result = await ai_provider.generate("What's the weather like today?")
    """


    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class Anthropic(BaseLLM):
    """
    Synchronous Anthropic Claude provider with enhanced error handling.

    This class interacts with Anthropic's Claude API synchronously to generate text completions.

    Attributes:
        api_key (str): The API key for authenticating with Anthropic.
        model (str): The model to use (default: "claude-3-opus-20240229").
        temperature (float): The temperature to control randomness (default: 0.0).
        max_tokens (int): The maximum number of tokens to generate (default: 2000).

    Methods:
        generate(prompt: str) -> str:
            Synchronously generates a response from Anthropic Claude based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = Anthropic(api_key="your-api-key")
        # result = ai_provider.generate("What's the weather like today?")
    """


    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class AsyncOllama(BaseLLM):
    """
    Asynchronous Ollama provider with enhanced error handling and streaming support.

    This class interacts with Ollama's API asynchronously to generate text completions.

    Attributes:
        model (str): The model to use (default: "llama2").
        host (str): The host for the Ollama API (default: "http://localhost:11434").

    Methods:
        generate(prompt: str) -> str:
            Asynchronously generates a response from Ollama based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = AsyncOllama(model="llama2", host="http://localhost:11434")
        # result = await ai_provider.generate("What's the weather like today?")
    """


    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        from ollama import AsyncClient

        self.client = AsyncClient(host=host)
        self.model = model

    async def generate(self, prompt: str) -> str:
        """
        Generates a response from the Ollama model asynchronously.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            Exception: If the generation fails.
        """
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
            )
            result = response["response"].strip()
            return result
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise


class Ollama(BaseLLM):
    """
    Synchronous Ollama provider with enhanced error handling and streaming support.

    This class interacts with Ollama's API synchronously to generate text completions.

    Attributes:
        model (str): The model to use (default: "llama2").
        host (str): The host for the Ollama API (default: "http://localhost:11434").

    Methods:
        generate(prompt: str) -> str:
            Synchronously generates a response from Ollama based on the provided prompt.

    Example:
        # To use this provider:
        # ai_provider = Ollama(model="llama2", host="http://localhost:11434")
        # result = ai_provider.generate("What's the weather like today?")
    """

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        from ollama import Client

        self.client = Client(host=host)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the Ollama model synchronously.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            Exception: If the generation fails.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
            )
            result = response["response"].strip()
            return result
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise


class AsyncNerdTokenApi(BaseLLM):
    """
    Asynchronous NerdToken API provider.

    This class interacts with NerdToken API to generate text completions asynchronously.

    Attributes:
        api_key (str): The API key for authenticating with NerdToken API.
        model (str): The model to use.

    Methods:
        generate(prompt: str, system_prompt: str, max_tokens: int, temperature: float, stream: bool, presence_penalty: float, frequency_penalty: float, top_p: float) -> str:
            Asynchronously generates a response from NerdToken API based on the provided prompt and settings.

    Example:
        # To use this provider:
        # ai_provider = AsyncNerdTokenApi(api_key="your-api-key", model="your-model")
        # result = await ai_provider.generate("Provide a summary of the latest news.", system_prompt="Be concise.", max_tokens=100)
    """


    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
        max_tokens: int = 4000,
        temperature: float = 0.3,
        stream: bool = True,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        top_p: float = 1,
    ) -> str:
        url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": prompt, "role": "user"},
            ],
            "model": self.model,
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                answer_data = response.json()
                generated_text = answer_data.get("text_message", "")
                return generated_text
            else:
                logger.error(
                    f"Error From Indox API: {response.status_code}, {response.text}"
                )
                raise Exception(
                    f"Error From Indox API: {response.status_code}, {response.text}"
                )


# class NerdTokenApi(BaseLLM):
#     """Synchronous Nerd Token API"""
#
#     def __init__(self, api_key: str, model: str):
#         self.api_key = api_key
#         self.model = model
#
#     def generate(
#         self,
#         prompt: str,
#         system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
#         max_tokens: int = 4000,
#         temperature: float = 0.3,
#         stream: bool = False,
#         presence_penalty: float = 0,
#         frequency_penalty: float = 0,
#         top_p: float = 1,
#     ) -> str:
#         url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
#         headers = {
#             "accept": "application/json",
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#
#         data = {
#             "frequency_penalty": frequency_penalty,
#             "max_tokens": max_tokens,
#             "messages": [
#                 {"content": system_prompt, "role": "system"},
#                 {"content": prompt, "role": "user"},
#             ],
#             "model": self.model,
#             "presence_penalty": presence_penalty,
#             "stream": stream,
#             "temperature": temperature,
#             "top_p": top_p,
#         }
#
#         response = requests.post(url, headers=headers, json=data)
#         print(response)
#         if response.status_code == 200:
#             answer_data = response.json()
#             generated_text = answer_data.get("text_message", "")
#             return generated_text
#         else:
#             logger.error(
#                 f"Error From Nerd Token API: {response.status_code}, {response.text}"
#             )
#             raise Exception(
#                 f"Error From Nerd Token API: {response.status_code}, {response.text}"
#             )

class NerdTokenApi(BaseLLM):
    """
    Synchronous NerdToken API provider.

    This class interacts with NerdToken API to generate text completions synchronously.

    Attributes:
        api_key (str): The API key for authenticating with NerdToken API.
        model (str): The model to use.

    Methods:
        generate(prompt: str, system_prompt: str, max_tokens: int, temperature: float, stream: bool, presence_penalty: float, frequency_penalty: float, top_p: float) -> str:
            Synchronously generates a response from NerdToken API based on the provided prompt and settings.

    Example:
        # To use this provider:
        # ai_provider = NerdTokenApi(api_key="your-api-key", model="your-model")
        # result = ai_provider.generate("Provide a summary of the latest news.", system_prompt="Be concise.", max_tokens=100)
    """


    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(
            self,
            prompt: str,
            system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
            max_tokens: int = 4000,
            temperature: float = 0.3,
            stream: bool = False,
            presence_penalty: float = 0,
            frequency_penalty: float = 0,
            top_p: float = 1,
    ) -> str:
        url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {"content": prompt, "role": "user"},
                {"content": system_prompt, "role": "system"},
            ],
            "model": self.model,
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data["choices"][0]["message"]["content"]
            return generated_text
        else:
            print(f"Error From Nerd Token API: {response.status_code}, {response.text}")
            raise Exception(
                f"Error From Nerd Token API: {response.status_code}, {response.text}"
            )


class AsyncVLLM(BaseLLM):
    """Asynchronous vLLM provider with enhanced error handling."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "vllm-default",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        base_url: str = "http://localhost:8000/v1",
    ):
        import aiohttp

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    async def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/completions", json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["text"]
                    else:
                        logger.error(
                            f"vLLM generation failed with status {response.status}"
                        )
                        raise Exception(
                            f"vLLM generation failed with status {response.status}"
                        )
            except Exception as e:
                logger.error(f"vLLM generation failed: {e}")
                raise


class VLLM(BaseLLM):
    """Synchronous vLLM provider with enhanced error handling."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "vllm-default",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        base_url: str = "http://localhost:8000/v1",
    ):
        import requests

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/completions", json=payload, headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["text"]
            else:
                logger.error(
                    f"vLLM generation failed with status {response.status_code}"
                )
                raise Exception(
                    f"vLLM generation failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise


class IndoxApi(BaseLLM):
    """Synchronous API"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
        max_tokens: int = 4000,
        temperature: float = 0.3,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        top_p: float = 1,
    ) -> str:
        url = "http://5.78.55.161/api/chat_completion/generate/"
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": prompt, "role": "user"},
            ],
            "model": "gpt-4o-mini",
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(url, headers=headers, json=data)
        print(response)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            logger.error(
                f"Error From Nerd Token API: {response.status_code}, {response.text}"
            )
            raise Exception(
                f"Error From Nerd Token API: {response.status_code}, {response.text}"
            )