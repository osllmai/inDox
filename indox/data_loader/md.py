import os
from indox.core.document_object import Document
from typing import List

class Md:

def Md(file_path: str):
    """
    Load a Markdown file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the Markdown file to be loaded.

    Methods:
    - load_file(): Loads the Markdown file and returns a list of `Document` objects, each containing
                   the text content of the file and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the Markdown file.

    Notes:
    - Metadata includes the file path and a fixed page number of 1.
    """


    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            metadata_dict = {
                'source': self.file_path,
                'page': 1,
            }

            document = Document(page_content=text, **metadata_dict)

            return [document]
        except Exception as e:
            raise RuntimeError(f"Error loading Markdown file: {e}")

