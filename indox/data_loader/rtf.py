from indox.core.document_object import Document
from typing import List
import os


class Rtf:
    """
    Load an RTF file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the RTF file to be loaded.

    Methods:
    - load_file(): Loads the RTF file and returns a list containing a single `Document` object with
                   the text content of the file and associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the RTF file.

    Notes:
    - Metadata includes only 'source' and page number.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        import pyth.plugins.rtf15.reader as rtf_reader
        import pyth.plugins.plaintext.writer as plaintext_writer

        try:
            with open(self.file_path, 'rb') as f:
                doc = rtf_reader.read(f)
                text = plaintext_writer.write(doc).getvalue()

                # metadata_dict = {
                #     'source': os.path.abspath(self.file_path),
                #     'page': 1
                # }
                #
                # return [Document(page_content=text, metadata=metadata_dict)]
                return text
        except Exception as e:
            raise RuntimeError(f"Error loading RTF file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
