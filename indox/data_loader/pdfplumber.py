from indox.core.document_object import Document
from typing import List
import os


class PdfPlumber:
    """
    Load a PDF file and extract its text and metadata using the pdfplumber library.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Methods:
    - load_file(): Loads the PDF file and returns a list of `Document` objects, each containing the text
                   content of a page and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: If there is an error reading the PDF file or extracting text.

    Notes:
    - Uses the `pdfplumber` library to read and extract text from the PDF.
    - Metadata includes the file path and page number.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        import pdfplumber
        try:
            with pdfplumber.open(self.file_path) as pdf:
                documents = []
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        raise RuntimeError(f"Error extracting text from page {i}. Details: {e}")

                    metadata_dict = {
                        'source': self.file_path,
                        'page': i
                    }

                    document = Document(page_content=text, metadata=metadata_dict)
                    documents.append(document)

                return documents
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while processing PDF file: {self.file_path}. Details: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
