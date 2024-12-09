import os
import warnings
from typing import Callable, Dict, Optional, Union
from importlib.metadata import version
from packaging.version import parse
from pydantic.v1 import Field, root_validator

from indoxRag.embeddings import OpenAiEmbedding
from indoxRag.embeddings.utils import get_from_dict_or_env


def is_openai_v1() -> bool:
    """Return whether OpenAI API is v1 or more."""
    _version = parse(version("openai"))
    return _version.major >= 1


class AzureOpenAIEmbeddings(OpenAiEmbedding):
    """`Azure OpenAI` Embeddings API."""

    azure_endpoint: Union[str, None] = None
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment: Optional[str] = Field(default=None, alias="azure_deployment")
    """A model deployment. 

        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    openai_api_key: Union[str, None] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Union[str, None] = None
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more: 
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every request.
    """
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    validate_base_url: bool = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        # Check OPENAI_KEY for backwards compatibility.
        # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
        # other forms of azure credentials.
        values["openai_api_key"] = (
            values.get("openai_api_key")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        values["openai_api_base"] = values.get("openai_api_base") or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_version"] = values.get("openai_api_version") or os.getenv(
            "OPENAI_API_VERSION", default="2023-05-15"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        values["openai_organization"] = (
            values.get("openai_organization")
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["azure_endpoint"] = values.get("azure_endpoint") or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        values["azure_ad_token"] = values.get("azure_ad_token") or os.getenv(
            "AZURE_OPENAI_AD_TOKEN"
        )
        # Azure OpenAI embedding models allow a maximum of 2048 texts
        # at a time in each batch
        # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        values["chunk_size"] = min(values["chunk_size"], 2048)
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if is_openai_v1():
            # For backwards compatibility. Before openai v1, no distinction was made
            # between azure_endpoint and base_url (openai_api_base).
            openai_api_base = values["openai_api_base"]
            if openai_api_base and values["validate_base_url"]:
                if "/openai" not in openai_api_base:
                    values["openai_api_base"] += "/openai"
                    warnings.warn(
                        "As of openai>=1.0.0, Azure endpoints should be specified via "
                        f"the `azure_endpoint` param not `openai_api_base` "
                        f"(or alias `base_url`). Updating `openai_api_base` from "
                        f"{openai_api_base} to {values['openai_api_base']}."
                    )
                if values["deployment"]:
                    warnings.warn(
                        "As of openai>=1.0.0, if `deployment` (or alias "
                        "`azure_deployment`) is specified then "
                        "`openai_api_base` (or alias `base_url`) should not be. "
                        "Instead use `deployment` (or alias `azure_deployment`) "
                        "and `azure_endpoint`."
                    )
                    if values["deployment"] not in values["openai_api_base"]:
                        warnings.warn(
                            "As of openai>=1.0.0, if `openai_api_base` "
                            "(or alias `base_url`) is specified it is expected to be "
                            "of the form "
                            "https://example-resource.azure.openai.com/openai/deployments/example-deployment. "  # noqa: E501
                            f"Updating {openai_api_base} to "
                            f"{values['openai_api_base']}."
                        )
                        values["openai_api_base"] += (
                            "/deployments/" + values["deployment"]
                        )
                    values["deployment"] = None
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def post_init_validator(cls, values: Dict) -> Dict:
        """Validate that the base url is set."""
        import openai

        if is_openai_v1():
            client_params = {
                "api_version": values["openai_api_version"],
                "azure_endpoint": values["azure_endpoint"],
                "azure_deployment": values["deployment"],
                "api_key": values["openai_api_key"],
                "azure_ad_token": values["azure_ad_token"],
                "azure_ad_token_provider": values["azure_ad_token_provider"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
                "default_headers": values["default_headers"],
                "default_query": values["default_query"],
                "http_client": values["http_client"],
            }
            values["client"] = openai.AzureOpenAI(**client_params).embeddings
            values["async_client"] = openai.AsyncAzureOpenAI(**client_params).embeddings
        else:
            values["client"] = openai.Embedding
        return values

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"