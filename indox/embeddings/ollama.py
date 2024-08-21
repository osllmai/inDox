from typing import Any, Dict, List, Mapping, Optional

import requests
from indox.core import Embeddings
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class OllamaEmbeddings(Embeddings):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Attributes:
        base_url: The base URL where the model is hosted.
        model: The name of the model to use.
        embed_instruction: Instruction used to embed documents.
        query_instruction: Instruction used to embed the query.
        mirostat: Optional integer to enable Mirostat sampling for controlling perplexity.
        mirostat_eta: Optional float to influence how quickly the algorithm responds to feedback.
        mirostat_tau: Optional float to control the balance between coherence and diversity.
        num_ctx: Optional integer to set the size of the context window used to generate the next token.
        num_gpu: Optional integer to set the number of GPUs to use.
        num_thread: Optional integer to set the number of threads to use during computation.
        repeat_last_n: Optional integer to set how far back the model looks to prevent repetition.
        repeat_penalty: Optional float to set how strongly to penalize repetitions.
        temperature: Optional float to set the temperature of the model.
        stop: Optional list of strings to set the stop tokens to use.
        tfs_z: Optional float to reduce the impact of less probable tokens from the output.
        top_k: Optional integer to reduce the probability of generating nonsense.
        top_p: Optional float to work together with top-k for diversity in text generation.
        show_progress: Boolean to determine whether to show a progress bar.
        headers: Optional dictionary for additional headers to pass to the endpoint.
    """

    base_url: str = "http://localhost:11434"
    model: str = "llama2"

    embed_instruction: str = "passage: "
    query_instruction: str = "query: "

    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    tfs_z: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    show_progress: bool = False
    headers: Optional[dict] = None

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        }

    model_kwargs: Optional[dict] = None

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    class Config:
        extra = "forbid"

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from the API.

        Args:
            input: The input string to process.

        Returns:
            The embedding as a list of floats.

        Raises:
            ValueError: If there is an issue with the API request or response.
        """
        headers = {
            "Content-Type": "application/json",
            **(self.headers or {}),
        }

        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                headers=headers,
                json={"model": self.model, "prompt": input, **self._default_params},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                f"Error raised by inference API HTTP code: {res.status_code}, {res.text}"
            )
        try:
            return res.json()["embedding"]
        except (requests.exceptions.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

    def _embed(self, input: List[str]) -> List[List[float]]:
        """Embed a list of inputs using Ollama.

        Args:
            input: A list of strings to embed.

        Returns:
            A list of embeddings, one for each input string.
        """
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="OllamaEmbeddings")
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._process_emb_response(prompt) for prompt in iter_]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [f"{self.embed_instruction}{text}" for text in texts]
        return self._embed(instruction_pairs)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using an Ollama deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = f"{self.query_instruction}{text}"
        return self._embed([instruction_pair])[0]
