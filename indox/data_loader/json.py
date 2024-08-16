# json_loader.py
import json
from indox.core.document_object import Document
from typing import List, Dict


def Json(file_path: str) -> List[Document]:
    """
    Load a JSON file and return its content as a list of `Document` objects.

    Parameters
    ----------
    file_path : str
        Path to the JSON file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing a JSON key-value pair and associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            documents = []

            # Create metadata
            metadata_dict = {
                'file_name': file_path,
                'num_entries': len(json_data),
                # Add more metadata as needed
            }

            for key, value in json_data.items():
                # Convert key-value pair to a string for storage in Document
                content = f"{key}: {value}"

                # Create a Document for each key-value pair
                document = Document(page_content=content, key_name=key, **metadata_dict)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")
