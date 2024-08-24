import re
from typing import List

class CharacterTextSplitter:
    """
    A class for splitting text into chunks based on a specified separator.

    This class implements an algorithm to split text into chunks of a specified size,
    with an optional overlap between chunks. It uses a single separator to determine
    where to split the text.

    Attributes:
        separator (str): The string used to split the text.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        length_function (callable): A function used to calculate the length of text.

    """

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        """
        Initialize the CharacterTextSplitter.

        Args:
            separator (str): The string used to split the text. Defaults to "\n\n".
            chunk_size (int): The maximum size of each chunk. Defaults to 4000.
            chunk_overlap (int): The number of characters to overlap between chunks. Defaults to 200.
            length_function (callable): A function used to calculate the length of text. Defaults to len.
        """
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text: str) -> List[str]:
        """
        Split incoming text and return chunks.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        splits = self._split_text(text)

        documents = self._merge_splits(splits)

        return documents

    def _split_text(self, text: str) -> List[str]:
        """
        Split incoming text by separator.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of text splits.
        """
        splits = re.split(f"({re.escape(self.separator)})", text)
        return [s for s in splits if s != ""]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits into chunks of the specified size.

        Args:
            splits (List[str]): The list of text splits to be merged.

        Returns:
            List[str]: A list of merged text chunks.
        """
        separator_len = self.length_function(self.separator)

        docs = []
        current_doc = []
        total = 0

        for split in splits:
            _len = self.length_function(split)

            if total + _len + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = self._join_docs(current_doc)
                    if doc:
                        docs.append(doc)

                    # Keep the overlap
                    overlap = self._get_overlap(current_doc)
                    current_doc = overlap
                    total = sum(self.length_function(s) for s in overlap)

            current_doc.append(split)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc = self._join_docs(current_doc)
            if doc:
                docs.append(doc)

        return docs

    def _join_docs(self, docs: List[str]) -> str:
        """
        Join the document parts using the separator.

        Args:
            docs (List[str]): The list of document parts to be joined.

        Returns:
            str: The joined text, or None if the result is empty.
        """
        text = self.separator.join(docs).strip()
        return text if text else None

    def _get_overlap(self, current_doc: List[str]) -> List[str]:
        """
        Retrieve overlap based on the chunk_overlap.

        Args:
            current_doc (List[str]): The current document parts.

        Returns:
            List[str]: A list of document parts that form the overlap for the next chunk.
        """
        overlap_length = 0
        overlap = []
        for part in reversed(current_doc):
            if overlap_length + self.length_function(part) > self.chunk_overlap:
                break
            overlap.insert(0, part)
            overlap_length += self.length_function(part)
        return overlap