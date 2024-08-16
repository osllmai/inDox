# openpyxl.py
import pandas as pd
from openpyxl import load_workbook
from indox.core.document_object import Document


def OpenPyXl(file_path):
    """
    Load an Excel file and extract its data and metadata.

    Parameters
    ----------
    file_path : str
        Path to the Excel file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of a sheet
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the Excel file.
    """
    try:
        workbook = load_workbook(file_path, read_only=True)
        properties = workbook.properties

        # Extract metadata into a dictionary
        metadata_dict = {
            "author": properties.author,
            "title": properties.title,
            "subject": properties.subject,
            "keywords": properties.keywords,
            "created": properties.created,
            "modified": properties.modified,
            "last_modified_by": properties.lastModifiedBy,
            "category": properties.category,
            "comments": properties.comments,
            "company": properties.company,
            "manager": properties.manager
        }

        # Load the actual data
        excel_data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets

        documents = []
        for sheet_name, data in excel_data.items():
            text_content = data.to_string(index=False)
            document = Document(page_content=text_content, sheet_name=sheet_name, **metadata_dict)
            documents.append(document)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading Excel file: {e}")
