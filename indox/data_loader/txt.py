from indox.core.document_object import Document
import os

class Txt:
    """
    Load a text file and return its content and metadata.

    Parameters:
    - file_path (str): Path to the text file to be loaded.

    Methods:
    - load_file(): Loads the text file and returns a list containing a single `Document` object with
                   the text content of the file and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the text file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, 'r') as f:
                text = f.read()

            # Metadata extraction
            metadata_dict = {
                'source': os.path.basename(self.file_path),
                'page': 1
            }

            document = Document(page_content=text, **metadata_dict)

            return [document]
        except Exception as e:
            raise RuntimeError(f"Error loading text file: {e}")
