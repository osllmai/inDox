from indox.core.document_object import Document
from typing import List
import os


def PdfPlumber(file_path: str) -> List[Document]:
    """
    Load a PDF file and extract its text and metadata using pdfplumber.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of a page
      and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: If there is an error reading the PDF file or extracting text.

    Notes:
    - Uses the `pdfplumber` library to read and extract text from the PDF.
    - Metadata is directly retrieved from the `pdfplumber` object and used to populate the `Document` objects.
    """
    import pdfplumber

    file_path = os.path.abspath(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:
            documents = []
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                except Exception as e:
                    raise RuntimeError(f"Error extracting text from page {i}. Details: {e}")

                # Metadata structure
                metadata_dict = {
                    'source': file_path,
                    'page': i
                }

                document = Document(page_content=text, metadata=metadata_dict)
                documents.append(document)

            return documents
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while processing PDF file: {file_path}. Details: {e}")
