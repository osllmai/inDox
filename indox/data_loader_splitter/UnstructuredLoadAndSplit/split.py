from loguru import logger
import sys
from indox.data_loader_splitter.UnstructuredLoadAndSplit.loader import create_documents_unstructured
from unstructured.chunking.title import chunk_by_title
from indox.vector_stores.utils import filter_complex_metadata
from typing import List, Tuple, Optional, Any, Dict
from indox.core import Document
# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")

def get_chunks_unstructured(file_path, chunk_size, remove_sword, splitter):
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

        # Create initial document elements using the unstructured library
        elements = create_documents_unstructured(file_path)

        if splitter:
            logger.info("Using custom splitter")
            text = ""
            for el in elements:
                text += el.text

            documents = splitter(text=text, max_tokens=chunk_size)
        else:
            logger.info("Using title-based chunking")
            # Split elements based on the title and the specified max characters per chunk
            elements = chunk_by_title(elements, max_characters=chunk_size)
            documents = []

            # Convert each element into a `Document` object with relevant metadata
            for element in elements:
                metadata = element.metadata.to_dict()
                del metadata["languages"]  # Remove unnecessary metadata field

                for key, value in metadata.items():
                    if isinstance(value, list):
                        value = str(value)
                    metadata[key] = value

                if remove_sword:
                    from indox.data_loader_splitter.utils.clean import remove_stopwords
                    element.text = remove_stopwords(element.text)

                # documents.append(Document(page_content=element.text, metadata=**metadata))
                documents.append(Document(page_content=element.text.replace("\n", ""), **metadata))

            # Filter and sanitize complex metadata
            documents = filter_complex_metadata(documents=documents)

        logger.info("Completed chunking process")
        return documents

    except Exception as e:
        logger.error(f"Failed at step with error: {e}")
        raise


class UnstructuredLoadAndSplit:
    def __init__(self, file_path: str, remove_sword: bool = False, max_chunk_size: int = 500, splitter=None):
        """
        Initialize the UnstructuredLoadAndSplit class.

        Parameters:
        - file_path (str): The path to the file containing unstructured data.
        - max_chunk_size (int): The maximum size (in characters) for each chunk.
        - remove_sword (bool): Whether to remove stopwords from the text.
        - splitter: The splitter to use for splitting the document.
        """
        try:
            self.file_path = file_path
            self.remove_sword = remove_sword
            self.max_chunk_size = max_chunk_size
            self.splitter = splitter
            logger.info("UnstructuredLoadAndSplit initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing UnstructuredLoadAndSplit: {e}")
            raise

    def load_and_chunk(self) -> List['Document']:
        """
        Split an unstructured document into chunks.

        Returns:
        - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
        """
        try:
            logger.info("Getting all documents")
            docs = get_chunks_unstructured(self.file_path, self.max_chunk_size, self.remove_sword, self.splitter)
            logger.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logger.error(f"Error in get_all_docs: {e}")
            raise
