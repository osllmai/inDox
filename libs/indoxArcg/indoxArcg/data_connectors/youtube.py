from indoxArcg.data_connectors.utils import Document
from typing import Any, List


class YoutubeTranscriptReader:
    """YouTube Transcript reader.

    This class retrieves transcripts from YouTube videos using the `youtube_transcript_api`
    package. It processes a list of YouTube video links and fetches the transcript text for each video.

    Attributes:
        languages (tuple): Tuple of language codes to fetch transcripts in. Defaults to English ("en").
    """

    languages: tuple = ("en",)

    @classmethod
    def class_name(cls) -> str:
        """Get the name of the class.

        Returns:
            str: The name of the class, "YoutubeTranscriptReader".
        """
        return "YoutubeTranscriptReader"

    def load_data(
        self, ytlinks: List[str], **load_kwargs: Any
    ) -> List[Document] | Document:
        """Load transcripts from the provided YouTube video links.

        Args:
            ytlinks (List[str]): List of YouTube video links from which transcripts are to be fetched.
            **load_kwargs: Additional keyword arguments to pass to `YouTubeTranscriptApi.get_transcript`.

        Returns:
            List[Document]: A list of Document instances, each containing the transcript and metadata
            for a YouTube video.

        Raises:
            ImportError: If the `youtube_transcript_api` package is not installed.
            ValueError: If the YouTube link format is invalid and cannot extract the video ID.
            Exception: For other unexpected errors that may occur while fetching transcripts.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "`youtube_transcript_api` package not found. Please install it using `pip install youtube-transcript-api`."
            )

        documents = []
        for link in ytlinks:
            try:
                video_id = self._extract_video_id(link)
                srt = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=self.languages, **load_kwargs
                )
                transcript = "\n".join(chunk["text"] for chunk in srt)
                metadata = {
                    "video_id": video_id,
                    "link": link,
                    "language": self.languages,
                }
                documents.append(
                    Document(source="YouTube", content=transcript, metadata=metadata)
                )
            except ValueError as e:
                print(f"ValueError: {e}")
            except Exception as e:
                print(f"Unexpected error while processing link '{link}': {e}")
        if len(documents) == 1:
            return documents[0]

        return documents

    def load_content(self, ytlinks: List[str], **load_kwargs: Any) -> List[str] | str:
        """Load transcripts from the provided YouTube video links.

        Args:
            ytlinks (List[str]): List of YouTube video links from which transcripts are to be fetched.
            **load_kwargs: Additional keyword arguments to pass to `YouTubeTranscriptApi.get_transcript`.

        Returns:
            List[str]: A list of strings, each containing the transcript for a YouTube video.

        Raises:
            ImportError: If the `youtube_transcript_api` package is not installed.
            ValueError: If the YouTube link format is invalid and cannot extract the video ID.
            Exception: For other unexpected errors that may occur while fetching transcripts.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "`youtube_transcript_api` package not found. Please install it using `pip install youtube-transcript-api`."
            )

        transcripts = []
        for link in ytlinks:
            try:
                video_id = self._extract_video_id(link)
                srt = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=self.languages, **load_kwargs
                )
                transcript = "\n".join(chunk["text"] for chunk in srt)
                transcripts.append(transcript)
            except ValueError as e:
                print(f"ValueError: {e}")
            except Exception as e:
                print(f"Unexpected error while processing link '{link}': {e}")
        if len(transcripts) == 1:
            return transcripts[0]

        return transcripts

    def _extract_video_id(self, link: str) -> str:
        """Extract the video ID from a YouTube link.

        Args:
            link (str): The YouTube video link.

        Returns:
            str: The extracted video ID.

        Raises:
            ValueError: If the video ID cannot be extracted from the link.
        """
        try:
            video_id = link.split("?v=")[-1]
            if not video_id:
                raise ValueError(
                    "Invalid YouTube link format. Unable to extract video ID."
                )
            return video_id
        except IndexError:
            raise ValueError("Invalid YouTube link format. Unable to extract video ID.")
