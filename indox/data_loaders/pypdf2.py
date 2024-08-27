from typing import List
from indox.vector_stores.utils import filter_complex_metadata
from indox.core import Document


class PyPdf2:
    """
    Load a PDF file and extract its text and metadata.

    Parameters:
    - pdf_path (str): The path to the PDF file to be loaded.

    Methods:
    - load(): Reads the PDF file, extracts text, and creates a list of `Document` objects with metadata.

    Returns:
    - List[Document]: A list containing `Document` objects with the text content of each page and associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the PDF file.
    - RuntimeError: For any other errors encountered during PDF processing.
    """

    def __init__(self, pdf_path: str):
        import PyPDF2

        self.pdf_path = pdf_path
        self.pages: List[Document] = []
        self.metadata = {}
        self.PyPDF2 = PyPDF2
        self.filter_complex_metadata = filter_complex_metadata

    def load(self) -> List[Document]:
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = self.PyPDF2.PdfReader(file)
                self.metadata = {key: value for key, value in reader.metadata.items()}

                filtered_pdf_reader = self.filter_complex_metadata([self])[0]
                self.metadata = filtered_pdf_reader.metadata

                self.pages = [
                    Document(page_content=page.extract_text(), **self.metadata)
                    for page in reader.pages
                ]

                return self.pages

        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.pdf_path}' does not exist.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("There was an error decoding the PDF file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the PDF file: {e}")


    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)

