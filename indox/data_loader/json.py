import json
from typing import List
from indox.core.document_object import Document
import os

class Json:
    """
    Load a JSON file and return its content as a list of `Document` objects.

    Parameters:
    - file_path (str): The path to the JSON file to be loaded.

    Methods:
    - load_file(): Loads the JSON file and returns a list of `Document` objects, each containing a JSON
                   key-value pair and associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the JSON file.

    Notes:
    - Metadata includes the file name and the number of entries in the JSON data.
    - Each JSON key-value pair is stored as a string in a separate `Document` object.
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                documents = []

                # Metadata for all entries
                metadata_dict = {
                    'source': self.file_path,
                    'num_entries': len(json_data),
                }

                for key, value in json_data.items():
                    content = f"{key}: {value}"
                    document_metadata = {
                        'source': self.file_path,
                        'key': key
                    }
                    document = Document(page_content=content, metadata=document_metadata)
                    documents.append(document)

                return documents
        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {e}")


    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            documents = []

            metadata_dict = {
                'source':  os.path.abspath(file_path),
                'num_entries': len(json_data),
            }

            for key, value in json_data.items():
                content = f"{key}: {value}"
                document = Document(metadata={'source': file_path, 'key': key}, page_content=content)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")
