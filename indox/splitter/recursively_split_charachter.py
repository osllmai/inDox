from typing import List, Optional


class RecursiveCharacterTextSplitter:
    """
    A class for splitting text into chunks recursively based on specified separators.

    This class implements a recursive algorithm to split text into chunks of a specified size,
    with an optional overlap between chunks. It uses a list of separators to determine where
    to split the text, starting with the first separator in the list and moving to the next
    if the current separator is not found in the text.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        separators (List[str]): A list of separators to use for splitting the text.
    """

    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            separators: Optional[List[str]] = None
    ):
        """
        Initialize the RecursiveCharacterTextSplitter.

        Args:
            chunk_size (int): The maximum size of each chunk. Defaults to 4000.
            chunk_overlap (int): The number of characters to overlap between chunks. Defaults to 200.
            separators (Optional[List[str]]): A list of separators to use for splitting the text.
                                              Defaults to ["\n\n", "\n", ". ", " ", ""].
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into chunks.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split the text using the given separators.

        Args:
            text (str): The text to be split.
            separators (List[str]): The list of separators to use for splitting.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        separator = separators[-1]
        for sep in separators:
            if sep in text:
                separator = sep
                break

        splits = self._split_with_separator(text, separator)

        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                if current_chunk:
                    chunk = self._merge_splits(current_chunk, separator)
                    chunks.append(chunk)

                    if self.chunk_overlap > 0:
                        overlap_splits = self._get_overlap(current_chunk)
                        current_chunk = overlap_splits
                        current_length = sum(len(s) for s in overlap_splits)
                    else:
                        current_chunk = []
                        current_length = 0

                if split_length > self.chunk_size:
                    if len(separators) > 1:
                        nested_chunks = self._split_text(split, separators[1:])
                        chunks.extend(nested_chunks)
                    else:
                        chunks.append(split)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [split]
                    current_length = split_length

        if current_chunk:
            chunk = self._merge_splits(current_chunk, separator)
            chunks.append(chunk)

        return chunks

    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Split the text using the given separator.

        Args:
            text (str): The text to be split.
            separator (str): The separator to use for splitting.

        Returns:
            List[str]: A list of split text pieces, with separators included except for the last piece.
        """
        if separator:
            splits = text.split(separator)
            return [s + separator for s in splits[:-1]] + [splits[-1]]
        return list(text)

    def _merge_splits(self, splits: List[str], separator: str) -> str:
        """
        Merge the splits back into a single string.

        Args:
            splits (List[str]): The list of split text pieces to merge.
            separator (str): The separator to use when joining the splits.

        Returns:
            str: The merged text.
        """
        return separator.join(splits).strip()

    def _get_overlap(self, current_chunk: List[str]) -> List[str]:
        """
        Calculate the overlap for the next chunk.

        Args:
            current_chunk (List[str]): The current chunk of text splits.

        Returns:
            List[str]: A list of splits that form the overlap for the next chunk.
        """
        overlap_length = 0
        overlap_splits = []
        for s in reversed(current_chunk):
            if overlap_length + len(s) > self.chunk_overlap:
                break
            overlap_splits.insert(0, s)
            overlap_length += len(s)
        return overlap_splits
