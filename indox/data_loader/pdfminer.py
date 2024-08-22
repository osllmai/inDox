from indox.core.document_object import Document
from typing import List
import os


class PdfMiner:
    """
    Load a PDF file and extract its text and metadata using pdfminer.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Methods:
    - load_file(): Loads the PDF file and returns a list of `Document` objects, each containing the text
                   content of a page and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - PDFSyntaxError: If there is a syntax-related issue in the PDF file.
    - RuntimeError: For any other errors encountered during PDF processing.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        from pdfminer.high_level import extract_pages
        from pdfminer.pdfparser import PDFSyntaxError
        try:
            documents = []
            try:
                with open(self.file_path, 'rb') as f:
                    # Extract pages using pdfminer
                    for i, page_layout in enumerate(extract_pages(f)):
                        text = ''.join([element.get_text() for element in page_layout if hasattr(element, 'get_text')])

                        metadata_dict = {
                            'source': self.file_path,
                            'page': i
                        }
                        document = Document(page_content=text, metadata=metadata_dict)
                        documents.append(document)

                return documents
            except PDFSyntaxError as e:
                raise PDFSyntaxError(f"Error loading PDF file (SyntaxError): {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error while processing PDF file: {self.file_path}. Details: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")


