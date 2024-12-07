from loguru import logger
import sys
from typing import List, Tuple, Optional, Any, Dict
from indox.core import Document
from indox.data_loader_splitter.utils.unstructured_utills import get_chunks_unstructured

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")



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
