from typing import List

from loguru import logger
import sys

from indox.core import Document
from indox.data_loader_splitter.utils.unstructured_utills import create_documents_unstructured, get_chunks_unstructured

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class Unstructured:
    def __init__(self, file_path: str):
        """
        Initialize the UnstructuredLoadAndSplit class.

        Parameters:
        - file_path (str): The path to the file containing unstructured data.

        """
        try:
            self.file_path = file_path
        except Exception as e:
            logger.error(f"Error initializing Unstructured: {e}")
            raise

    def load(self):

        elements = create_documents_unstructured(file_path=self.file_path)
        return elements

    def load_and_split(self, remove_stopwords: bool = False, max_chunk_size: int = 500, splitter=None) -> (
            List)['Document']:
        """
        Split an unstructured document into chunks.

        Returns:
        - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
        """
        try:
            logger.info("Getting all documents")
            docs = get_chunks_unstructured(self.file_path, max_chunk_size, remove_stopwords, splitter)
            logger.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logger.error(f"Error in get_all_docs: {e}")
            raise
