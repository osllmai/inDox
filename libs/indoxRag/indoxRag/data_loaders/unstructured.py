import importlib
from typing import List

from loguru import logger
import sys

from indoxRag.core import Document
from indoxRag.data_loaders.utils import convert_latex_to_md
from indoxRag.vector_stores.utils import filter_complex_metadata

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


def import_unstructured_partition(content_type):
    # Import appropriate partition function from the `unstructured` library
    module_name = f"unstructured.partition.{content_type}"
    module = importlib.import_module(module_name)
    partition_function_name = f"partition_{content_type}"
    prt = getattr(module, partition_function_name)
    return prt


def create_documents_unstructured(file_path):
    try:
        if file_path.lower().endswith(".pdf"):
            # Partition PDF with a high-resolution strategy
            from unstructured.partition.pdf import partition_pdf

            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                # infer_table_structure=True,
            )
            # Remove "References" and header elements
            reference_title = [
                el
                for el in elements
                if el.text == "References" and el.category == "Title"
            ][0]
            references_id = reference_title.id
            elements = [el for el in elements if el.metadata.parent_id != references_id]
            elements = [el for el in elements if el.category != "Header"]
        elif file_path.lower().endswith(".xlsx"):
            from unstructured.partition.xlsx import partition_xlsx

            elements_ = partition_xlsx(filename=file_path)
            elements = [el for el in elements_ if el.metadata.text_as_html is not None]
        elif file_path.lower().startswith("www") or file_path.lower().startswith(
            "http"
        ):
            from unstructured.partition.html import partition_html

            elements = partition_html(url=file_path)
        else:
            if file_path.lower().endswith(".tex"):
                file_path = convert_latex_to_md(latex_path=file_path)
            content_type = file_path.lower().split(".")[-1]
            if content_type == "txt":
                prt = import_unstructured_partition(content_type="text")
            else:
                prt = import_unstructured_partition(content_type=content_type)
            elements = prt(filename=file_path)
        return elements
    except AttributeError as ae:
        logger.error(f"Attribute error: {ae}")
        return ae
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return e


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
    from unstructured.chunking.title import chunk_by_title

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
                    from indoxRag.data_loader_splitter.utils.clean import (
                        remove_stopwords,
                    )

                    element.text = remove_stopwords(element.text)

                # documents.append(Document(page_content=element.text, metadata=**metadata))
                documents.append(
                    Document(page_content=element.text.replace("\n", ""), **metadata)
                )

            # Filter and sanitize complex metadata
            documents = filter_complex_metadata(documents=documents)

        logger.info("Completed chunking process")
        return documents

    except Exception as e:
        logger.error(f"Failed at step with error: {e}")
        raise


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

    def load_and_split(
        self, remove_stopwords: bool = False, max_chunk_size: int = 500, splitter=None
    ) -> (List)["Document"]:
        """
        Split an unstructured document into chunks.

        Returns:
        - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
        """
        try:
            logger.info("Getting all documents")
            docs = get_chunks_unstructured(
                self.file_path, max_chunk_size, remove_stopwords, splitter
            )
            logger.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logger.error(f"Error in get_all_docs: {e}")
            raise
