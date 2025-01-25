# def semantic_text_splitter(text, max_tokens):
#     from semantic_text_splitter import TextSplitter
#     from tokenizers import Tokenizer
#     # Maximum number of tokens in a chunk
#     tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
#     splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)
#
#     chunks = splitter.chunks(text)
#     return chunks

from typing import List

class SemanticTextSplitter:
    """
    A class for splitting text into semantically meaningful chunks using a BERT tokenizer.

    This class uses the semantic_text_splitter library to split the input text
    into chunks that preserve semantic meaning, while ensuring that each chunk
    does not exceed the specified maximum number of tokens.

    Attributes:
        chunk_size (int): The maximum number of tokens allowed in each chunk.
        tokenizer (Tokenizer): The BERT tokenizer used for splitting.
        splitter (TextSplitter): The TextSplitter instance used for semantic splitting.
    """

    def __init__(
            self,
            chunk_size: int = 400,
            model_name: str = "bert-base-uncased"
    ):
        """
        Initialize the SemanticTextSplitter.

        Args:
            chunk_size (int): The maximum number of tokens allowed in each chunk. Defaults to 512.
            model_name (str): The name of the pre-trained model to use for the tokenizer. Defaults to "bert-base-uncased".
        """
        from semantic_text_splitter import TextSplitter
        from tokenizers import Tokenizer

        self.chunk_size = chunk_size
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        self.splitter = TextSplitter.from_huggingface_tokenizer(self.tokenizer, self.chunk_size)

    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into semantically meaningful chunks.

        Args:
            text (str): The input text to be split into chunks.

        Returns:
            List[str]: A list of text chunks, where each chunk is a string.
        """
        return self.splitter.chunks(text)


