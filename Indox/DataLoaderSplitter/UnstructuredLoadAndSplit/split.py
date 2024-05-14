from Indox.DataLoaderSplitter.utils.clean import remove_stopwords
from Indox.utils import create_documents_unstructured
from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List, Tuple, Optional, Any, Dict
from langchain_core.documents import Document


def get_chunks_unstructured(file_path, content_type, chunk_size, remove_sword, splitter):
    """
    Extract chunks from an unstructured document file using an unstructured data processing library.

    Parameters:
    - file_path (str): The path to the file containing unstructured data.
    - content_type (str): The type of content (e.g., "pdf", "html", etc.) to process.
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
        print("Starting processing...")

        # Create initial document elements using the unstructured library
        elements = create_documents_unstructured(file_path, content_type=content_type)
        if splitter:
            text = ""
            for el in elements:
                text += el.text

            documents = splitter(text=text, max_tokens=chunk_size)
        else:
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
                    element.text = remove_stopwords(element.text)

                documents.append(Document(page_content=element.text, metadata=metadata))

            # Filter and sanitize complex metadata
            documents = filter_complex_metadata(documents=documents)

        print("End Chunking process.")
        return documents

    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise


def UnstructuredLoadAndSplit(file_path: str, content_type: str, remove_sword: bool = False,
                             max_chunk_size: int = 500, splitter=None) -> (
        List)[Document]:
    """
    Split an unstructured document into chunks.

    Parameters:
    - file_path (str): The path to the file containing unstructured data.
    - max_chunk_size (int): The maximum size (in characters) for each chunk.
    - content_type (str): The type of content (e.g., "pdf", "html", etc.) to process.
    - remove_sword (bool): Whether to remove stopwords from the text.

    Returns:
    - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
    """
    return get_chunks_unstructured(file_path, content_type, max_chunk_size, remove_sword, splitter)
