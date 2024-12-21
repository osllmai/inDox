from typing import Any, Dict, Literal, Tuple, List, Type
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


class Document:
    """
    Class for storing a piece of text and associated metadata.

    Attributes:
        page_content (str): The content of the document (text).
        metadata (Dict[str, Any]): Metadata associated with the document, provided as keyword arguments.
        type (Literal["Document"]): The type of the object, always "Document".

    Methods:
        __repr__ (str): Returns a string representation of the Document, displaying its content and metadata.

    Example:
        doc = Document("Sample content", author="John Doe", year=2023)
        print(doc)  # Output: Document(page_content=Sample content, metadata={'author': 'John Doe', 'year': 2023})
    """


    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Initialize the Document with page content and arbitrary metadata."""
        self.page_content = page_content
        self.metadata: Dict[str, Any] = kwargs
        self.type: Literal["Document"] = "Document"

    def __repr__(self) -> str:
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"


def convert_latex_to_md(latex_path):
    """
    Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file to convert.

    Returns:
        str: The converted Markdown content, or `None` if there's an error during conversion.

    Example:
        markdown_content = convert_latex_to_md("example.tex")
        if markdown_content:
            print(markdown_content)  # Output: Converted markdown content from LaTeX
        else:
            print("Conversion failed")
    """

    import latex2markdown
    try:
        with open(latex_path, 'r') as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        logger.info(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        logger.error(f"Error during conversion: {e}")


def filter_complex_metadata(
        documents: List[Document],
        *,
        allowed_types: Tuple[Type, ...] = (str, bool, int, float),
) -> List[Document]:
    """
    Filter out metadata types that are not supported for a vector store.

    Args:
        documents (List[Document]): A list of `Document` objects to filter.
        allowed_types (Tuple[Type, ...], optional): A tuple of allowed metadata types (default is (str, bool, int, float)).

    Returns:
        List[Document]: A list of `Document` objects with filtered metadata.

    Example:
        documents = [
            Document("Content 1", author="John", year=2023, complex_data={"nested": "data"}),
            Document("Content 2", author="Jane", year=2024, valid_field=42)
        ]
        filtered_docs = filter_complex_metadata(documents)
        for doc in filtered_docs:
            print(doc.metadata)  # Output: {'author': 'John', 'year': 2023} (complex data is filtered out)
    """

    updated_documents = []
    for document in documents:
        filtered_metadata = {}
        for key, value in document.metadata.items():
            if not isinstance(value, allowed_types):
                continue
            filtered_metadata[key] = value

        document.metadata = filtered_metadata
        updated_documents.append(document)

    return updated_documents