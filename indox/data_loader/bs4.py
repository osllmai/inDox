from indox.core.document_object import Document
from typing import List

import os


class Bs4:
    """
    Load an HTML file and extract its text and minimal metadata.

    Parameters:
    - file_path (str): The path to the HTML file to be loaded.

    Methods:
    - load_file(): Reads the HTML file, extracts text, and creates a `Document` object with metadata.

    Returns:
    - List[Document]: A list containing a single `Document` object with the HTML text content and associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the HTML file.
    - RuntimeError: For any other errors encountered during HTML processing.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self):
        from bs4 import BeautifulSoup

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Error decoding HTML file: {self.file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading the HTML file: {self.file_path}. Details: {e}")

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
        except Exception as e:
            raise RuntimeError(f"Error parsing HTML content: {self.file_path}. Details: {e}")

        return text

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
