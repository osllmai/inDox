from typing import Any, Dict, List, Optional

from pydantic import Field, root_validator
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


class ClarifaiEmbeddings(Embeddings):
    """Clarifai embedding models.

    To use, you should have the `clarifai` python package installed, and the
    environment variable `CLARIFAI_PAT` set with your personal access token or pass it
    as a named parameter to the constructor.
    """

    model_url: Optional[str] = None
    model_id: Optional[str] = None
    model_version_id: Optional[str] = None
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    pat: Optional[str] = Field(default=None, exclude=True)
    token: Optional[str] = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)  #: :meta private:
    api_base: str = "https://api.clarifai.com"

    class Config:
        extra = "forbid"

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that we have all required info to access Clarifai platform
        and ensure the Python package exists in the environment."""
        try:
            from clarifai.client.model import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )

        required_fields = ["model_id", "pat"]
        for field in required_fields:
            if not values.get(field):
                raise ValueError(f"{field} is required for ClarifaiEmbeddings.")

        values["model"] = Model(
            url=values.get("model_url"),
            app_id=values.get("app_id"),
            user_id=values.get("user_id"),
            model_version=dict(id=values.get("model_version_id")),
            pat=values.get("pat"),
            token=values.get("token"),
            model_id=values.get("model_id"),
            base_url=values.get("api_base"),
        )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Clarifai's embedding models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            logger.warning("No texts provided for embedding.")
            return []

        from clarifai.client.input import Inputs

        input_obj = Inputs.from_auth_helper(self.model.auth_helper)
        batch_size = 32
        embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                input_batch = [
                    input_obj.get_text_input(input_id=str(id), raw_text=inp)
                    for id, inp in enumerate(batch)
                ]
                predict_response = self.model.predict(input_batch)
                embeddings.extend(
                    [
                        list(output.data.embeddings[0].vector)
                        for output in predict_response.outputs
                    ]
                )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection or timeout error during prediction: {e}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Clarifai's embedding models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            predict_response = self.model.predict_by_bytes(
                bytes(text, "utf-8"), input_type="text"
            )
            embeddings = [
                list(op.data.embeddings[0].vector) for op in predict_response.outputs
            ]
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection or timeout error during prediction: {e}")
            return []
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

        return embeddings[0] if embeddings else []
