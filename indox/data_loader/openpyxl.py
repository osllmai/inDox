import pandas as pd
from indox.core.document_object import Document
import os
from typing import List


class OpenPyXl:
    """
    Load an Excel file and extract its data and metadata.

    Parameters:
    - file_path (str): Path to the Excel file to be loaded.

    Methods:
    - load_file(): Loads the Excel file and returns a list of `Document` objects, each containing the text
                   content of a sheet and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the Excel file.

    Notes:
    - Metadata includes properties such as author, title, keywords, creation and modification dates, etc.
    - Each sheet in the Excel file is loaded into a separate `Document` object.
    - The data in each sheet is converted to a string for storage in the `Document` object.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        from openpyxl import load_workbook

        try:
            # Load workbook properties
            workbook = load_workbook(self.file_path, read_only=True)
            properties = workbook.properties

            # Metadata extraction
            metadata_dict = {
                "source": self.file_path,
                "page": 1
            }

            # Load the actual data
            excel_data = pd.read_excel(self.file_path, sheet_name=None)

            documents = []
            for sheet_name, data in excel_data.items():
                text_content = data.to_string(index=False)
                document = Document(page_content=text_content, sheet_name=sheet_name, **metadata_dict)
                documents.append(document)

            return documents

        except Exception as e:
            raise RuntimeError(f"Error loading Excel file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
