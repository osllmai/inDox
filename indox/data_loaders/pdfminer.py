from typing import List
from indox.vector_stores.utils import filter_complex_metadata
from indox.core import Document


class PdfMiner:
    """
    Load a PDF file and extract its text and metadata using pdfminer.

    Parameters:
    - pdf_path (str): The path to the PDF file to be loaded.

    Methods:
    - load(): Reads the PDF file, extracts text, and creates a list of `Document` objects with metadata.

    Returns:
    - List[Document]: A list containing `Document` objects with the text content of each page and associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: For any other errors encountered during PDF processing.
    """

    def __init__(self, pdf_path: str):
        from pdfminer.high_level import extract_text
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument

        self.pdf_path = pdf_path
        self.pages: List[Document] = []
        self.metadata = {}
        self.extract_text = extract_text
        self.PDFParser = PDFParser
        self.PDFDocument = PDFDocument
        self.filter_complex_metadata = filter_complex_metadata

    def load(self) -> List[Document]:
        try:
            page_texts = self.extract_text(self.pdf_path).split('\f')

            with open(self.pdf_path, 'rb') as file:
                parser = self.PDFParser(file)
                document = self.PDFDocument(parser)
                self.metadata = document.info[0] if document.info else {}

            filtered_pdf_reader = self.filter_complex_metadata([self])[0]
            self.metadata = filtered_pdf_reader.metadata

            self.pages = [
                Document(page_content=page_text.strip(), **self.metadata)
                for page_text in page_texts
            ]

            return self.pages

        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.pdf_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the PDF file: {e}")


    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)

