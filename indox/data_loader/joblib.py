import os
from indox.core.document_object import Document
from typing import List


class Joblib:
    """
    Load a PKL (Pickle) or Joblib file and extract its content.

    Parameters:
    - file_path (str): The path to the PKL or Joblib file to be loaded.

    Methods:
    - load_file(): Loads the file and returns a `Document` object containing the unpickled content and associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the file.

    Notes:
    - Metadata includes 'source' and page number.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self):
        import joblib

        try:
            content = joblib.load(self.file_path)

            # metadata_dict = {
            #     'source': self.file_path,
            #     'page': 1
            # }
            # document = Document(page_content=str(content), metadata=metadata_dict)
            #
            # return [document]
            return content
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
