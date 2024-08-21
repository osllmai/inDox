import pandas as pd
from indox.core.document_object import Document
import os


def OpenPyXl(file_path):
    """
    Load an Excel file and extract its data and metadata.

    Parameters:
    - file_path (str): Path to the Excel file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of a sheet
      and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the Excel file.

    Notes:
    - Metadata includes properties such as author, title, keywords, creation and modification dates, etc.
    - Each sheet in the Excel file is loaded into a separate `Document` object.
    - The data in each sheet is converted to a string for storage in the `Document` object.
    """

    from openpyxl import load_workbook

    try:
        workbook = load_workbook(file_path, read_only=True)
        properties = workbook.properties

        metadata_dict = {
            "source": os.path.abspath(file_path),
            "page": 1
        }

        # Load the actual data
        excel_data = pd.read_excel(file_path, sheet_name=None)

        documents = []
        for sheet_name, data in excel_data.items():
            text_content = data.to_string(index=False)
            document = Document(page_content=text_content, sheet_name=sheet_name, **metadata_dict)
            documents.append(document)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading Excel file: {e}")
