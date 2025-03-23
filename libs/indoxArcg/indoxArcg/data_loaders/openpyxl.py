import pandas as pd
from indoxArcg.core.document_object import Document
import os
from typing import List, Optional, Union


class OpenPyXl:
    """
    Load an Excel file and extract its data and metadata.

    Parameters:
    - file_path (str, optional): Path to the Excel file to be loaded.
    - sheet_names (List[str], optional): Specific sheets to load. If None, all sheets will be loaded.

    Methods:
    - load(file_path=None): Loads the Excel file and returns a list of `Document` objects, each containing 
                           the text content of a sheet and the associated metadata.
    - load_and_split(splitter, remove_stopwords=False): Loads the file, splits the content using the provided
                                                       splitter, and processes it.

    Raises:
    - RuntimeError: If there is an error in loading the Excel file.

    Notes:
    - Metadata includes properties such as author, title, keywords, creation and modification dates, etc.
    - Each sheet in the Excel file is loaded into a separate `Document` object.
    - The data in each sheet is converted to a string for storage in the `Document` object.
    """

    def __init__(self, file_path: Optional[str] = None, sheet_names: Optional[List[str]] = None):
        self.file_path = os.path.abspath(file_path) if file_path else None
        self.sheet_names = sheet_names

    def load(self, file_path: Optional[str] = None) -> List[Document]:
        from openpyxl import load_workbook

        # Use the file_path parameter if provided, otherwise use the one from initialization
        if file_path:
            self.file_path = os.path.abspath(file_path)
        
        if not self.file_path:
            raise ValueError("No file path provided. Either pass it during initialization or to the load method.")

        try:
            # Load workbook properties
            workbook = load_workbook(self.file_path, read_only=True)
            properties = workbook.properties

            # Metadata extraction
            metadata_dict = {"source": self.file_path, "page": 1}

            # Load the actual data based on sheet_names
            if self.sheet_names:
                excel_data = pd.read_excel(self.file_path, sheet_name=self.sheet_names)
            else:
                excel_data = pd.read_excel(self.file_path, sheet_name=None)

            documents = []
            for sheet_name, data in excel_data.items():
                text_content = data.to_string(index=False)
                document = Document(
                    page_content=text_content, sheet_name=sheet_name, **metadata_dict
                )
                documents.append(document)

            return documents

        except Exception as e:
            raise RuntimeError(f"Error loading Excel file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indoxArcg.data_loaders.utils import load_and_process_input

        return load_and_process_input(
            loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords
        )