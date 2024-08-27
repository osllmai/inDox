import os
import requests
from typing import List, Optional


class AI21SemanticTextSplitter:
    """
    A class for splitting text into semantic chunks using the AI21 API.

    This class provides functionality to split text into meaningful segments
    using AI21's semantic text segmentation API. It can also merge these
    segments into larger chunks if a chunk size is specified.
    """

    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            api_key: Optional[str] = None,
            api_host: str = "https://api.ai21.com/studio/v1",
            timeout_sec: Optional[float] = None,
            num_retries: int = 3,
    ):
        """
        Initialize the AI21SemanticTextSplitter.

        Args:
            chunk_size (int): The maximum size of each chunk. If 0, no merging is done.
            chunk_overlap (int): The number of characters to overlap between chunks.
            api_key (Optional[str]): The AI21 API key. If not provided, it will be read from the AI21_API_KEY environment variable.
            api_host (str): The base URL for the AI21 API.
            timeout_sec (Optional[float]): The timeout for API requests in seconds.
            num_retries (int): The number of times to retry failed API calls.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key or os.environ.get("AI21_API_KEY")
        self.api_host = api_host
        self.timeout_sec = timeout_sec
        self.num_retries = num_retries

        if not self.api_key:
            raise ValueError("AI21_API_KEY must be provided or set as an environment variable.")

    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into semantic segments.

        This method calls the AI21 API to perform semantic segmentation on the input text.
        If a chunk_size is specified, it also merges the segments into larger chunks.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text segments or chunks.
        """
        segments = self._call_segmentation_api(text)

        if self.chunk_size > 0:
            return self._merge_splits(segments)
        return segments

    def _call_segmentation_api(self, text: str) -> List[str]:
        """
        Call the AI21 segmentation API to split the text.

        This method makes a POST request to the AI21 API to perform semantic segmentation.
        It includes retry logic in case of failed API calls.

        Args:
            text (str): The input text to be segmented.

        Returns:
            List[str]: A list of segmented text pieces.

        Raises:
            Exception: If all API call attempts fail.
        """
        url = f"{self.api_host}/segmentation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "source": text,
            "sourceType": "TEXT"
        }

        for _ in range(self.num_retries):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=self.timeout_sec)
                response.raise_for_status()
                result = response.json()
                return [segment['segmentText'] for segment in result['segments']]
            except requests.RequestException as e:
                print(f"API call failed: {e}")

        raise Exception("Failed to call AI21 API after multiple retries")

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge the split segments into larger chunks.

        This method combines the split segments into chunks of size up to chunk_size.

        Args:
            splits (List[str]): The list of text segments to be merged.

        Returns:
            List[str]: A list of merged text chunks.
        """
        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            current_chunk += split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
