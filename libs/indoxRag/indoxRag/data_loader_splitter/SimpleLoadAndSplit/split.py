from typing import Optional, List, Tuple
from indoxRag.core import Document
from indoxRag.data_loader_splitter.SimpleLoadAndSplit.loader import create_document
from loguru import logger
import sys
from indoxRag.splitter import semantic_text_splitter

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


def get_chunks(file_path, chunk_size, remove_sword):
    """
    Extract chunks from an unstructured document file using an unstructured data processing library.

    Parameters:
    - file_path (str): The path to the file containing unstructured data.
    - chunk_size (int, optional): The maximum size (in characters) for each chunk. Defaults to 500.

    Returns:
    - list: A list of `Document` objects, each containing a portion of the original content with relevant metadata.

    Raises:
    - Exception: Any errors that occur during document processing or chunking.

    Notes:
    - The function uses a title-based chunking method to segment the unstructured data into logical parts.
    - Metadata is cleaned and filtered to ensure proper structure before being added to the `Document` objects.
    - The `filter_complex_metadata` function is used to simplify and sanitize metadata attributes.

    """
    try:
        logger.info("Starting processing")

        # Create initial document elements using the 'PyPDF2' library
        text = create_document(file_path)
        logger.info("Created initial document elements")

        # split text by using 'bert-base-uncased' model and split it into chunks
        texts = semantic_text_splitter(text, max_tokens=chunk_size)
        for i in range(len(texts)):
            texts[i] = texts[i].replace("\n", " ")

        # Optionally remove stopwords from the chunks
        if remove_sword:
            from indox.data_loader_splitter.utils.clean import remove_stopwords_chunk

            texts = remove_stopwords_chunk(texts)

        logger.info("Completed chunking process")
        return texts

    except Exception as e:
        logger.error("Failed at step with error: %s", e)
        raise


class SimpleLoadAndSplit:
    def __init__(
        self, file_path: str, remove_sword: bool = False, max_chunk_size: int = 500
    ):
        try:
            self.file_path = file_path
            self.remove_sword = remove_sword
            self.max_chunk_size = max_chunk_size
            logger.info("UnstructuredLoadAndSplit initialized successfully")
        except Exception as e:
            logger.error("Error initializing UnstructuredLoadAndSplit: %s", e)
            raise

    def load_and_chunk(self) -> List["Document"]:
        """
        Split an unstructured document into chunks.

        Returns:
        - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
        """
        try:
            docs = get_chunks(self.file_path, self.max_chunk_size, self.remove_sword)
            logger.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logger.error("Error in get_all_docs: %s", e)
            raise
