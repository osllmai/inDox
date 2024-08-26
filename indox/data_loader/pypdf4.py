from indox.core.document_object import Document
from typing import List
import os


class PyPdf4:
    """
    Load a PDF file and extract its text and metadata using PyPDF4.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Methods:
    - load_file(): Loads the PDF file and returns a list containing `Document` objects, each with the
                   text content of a page and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: If there is an error reading the PDF file or extracting text.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        import PyPDF4

        try:
            with open(self.file_path, 'rb') as f:
                try:
                    reader = PyPDF4.PdfFileReader(f)
                except PyPDF4.utils.PdfReadError as e:
                    raise RuntimeError(f"Error reading PDF file: {self.file_path}. Details: {e}")

                documents = []
                for i in range(reader.getNumPages()):
                    page = reader.getPage(i)
                    try:
                        text = page.extractText()
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
