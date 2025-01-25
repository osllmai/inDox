from typing import Any, Dict, List, Optional

from pydantic.v1 import root_validator
from indoxArcg.core import Embeddings
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


class GPT4AllEmbeddings(Embeddings):
    """GPT4All embedding models.

    To use, you should have the gpt4all python package installed.
    """

    model_name: Optional[str] = None
    n_threads: Optional[int] = None
    device: Optional[str] = "cpu"
    gpt4all_kwargs: Optional[Dict[str, Any]] = {}
    client: Any = None  #: :meta private:

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed and initialize the client."""
        try:
            from gpt4all import Embed4All

            model_name = values.get("model_name")
            if not model_name:
                raise ValueError("model_name must be provided for GPT4AllEmbeddings.")

            values["client"] = Embed4All(
                model_name=model_name,
                n_threads=values.get("n_threads"),
                device=values.get("device"),
                **(values.get("gpt4all_kwargs") or {}),
            )
        except ImportError:
            raise ImportError(
                "Could not import gpt4all library. "
                "Please install the gpt4all library to "
                "use this embedding model: pip install gpt4all"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using GPT4All.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            embeddings = [self.client.embed(text) for text in texts]
            return [list(map(float, e)) for e in embeddings]
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using GPT4All.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        try:
            return self.embed_documents([text])[0]
        except IndexError:
            logger.error("Failed to embed query: No embeddings were returned.")
            return []
