from indox.splitter import RecursiveCharacterTextSplitter
from typing import List, Any


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """
    A class for splitting Markdown-formatted text into chunks.

    This class extends RecursiveCharacterTextSplitter to specifically handle
    Markdown-formatted text, splitting along headings and other Markdown-specific
    separators.

    Attributes:
        Inherits all attributes from RecursiveCharacterTextSplitter.

    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a MarkdownTextSplitter.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.

        The initializer sets up Markdown-specific separators before calling the
        parent class initializer.
        """
        separators = [
            "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
            "\n```\n", "\n\n***\n\n", "\n\n---\n\n", "\n\n___\n\n",
            "\n\n", "\n", " ", ""
        ]
        super().__init__(separators=separators, **kwargs)

    def split_text(self, text: str) -> List[str]:
        """
        Preprocess and split the Markdown text into chunks.

        This method first preprocesses the text to standardize heading formats,
        then calls the parent class's split_text method to perform the actual splitting.

        Args:
            text (str): The Markdown-formatted text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        text = self._preprocess_text(text)
        return super().split_text(text)

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the Markdown text to standardize heading formats.

        This method converts alternative Markdown heading styles (using '=' and '-'
        underlines) to the standard '#' prefix style.

        Args:
            text (str): The original Markdown-formatted text.

        Returns:
            str: The preprocessed Markdown text with standardized heading formats.
        """
        lines = text.split('\n')
        for i in range(len(lines) - 1):
            if set(lines[i + 1]) == {'='}:
                lines[i] = '# ' + lines[i]
                lines[i + 1] = ''
            elif set(lines[i + 1]) == {'-'}:
                lines[i] = '## ' + lines[i]
                lines[i + 1] = ''
        return '\n'.join(lines)
