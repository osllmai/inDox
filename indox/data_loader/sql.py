from indox.core.document_object import Document
import os

class Sql:
    """
    Load an SQL file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the SQL file to be loaded.

    Methods:
    - load_file(): Loads the SQL file and returns a list containing a single `Document` object with
                   the text content of the file and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the SQL file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # metadata_dict = {
            #     'source': os.path.basename(self.file_path),
            #     'page': 1
            # }
            #
            # document = Document(page_content=text, **metadata_dict)
            #
            # return [document]
            return text
        except Exception as e:
            raise RuntimeError(f"Error loading SQL file: {e}")


