# csv.py
import csv
from indox.core.document_object import Document
from typing import List, Dict, Any


def Csv(file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Load a CSV file and return its content as a list of `Document` objects.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to be loaded.
    metadata : dict, optional
        Additional metadata to include in each `Document`. Default is None.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the CSV rows and associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the CSV file.
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            documents = []
            metadata_dict = {'file_name': file_path}

            # Add any additional metadata provided
            if metadata:
                metadata_dict.update(metadata)

            for i, row in enumerate(reader):
                # Convert row to a string for storage in Document
                row_content = ','.join(row)

                # Create a Document for each row
                document = Document(page_content=row_content, row_number=i + 1, **metadata_dict)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")
