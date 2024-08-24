import csv
from indox.core.document_object import Document
from typing import List, Dict, Any
import os


class Csv:
    """
    Load a CSV file and return its content as a list of `Document` objects.

    Parameters:
    - file_path (str): The path to the CSV file to be loaded.
    - metadata (dict, optional): Additional metadata to include in each `Document`. Default is None.

    Methods:
    - load_file(): Reads the CSV file and creates a list of `Document` objects with associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an issue decoding the CSV file.
    - RuntimeError: For any other errors encountered during CSV processing.
    """

    def __init__(self, file_path: str, metadata: Dict[str, Any] = None):
        self.file_path = os.path.abspath(file_path)
        self.metadata = metadata if metadata is not None else {}

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                num_rows = sum(1 for _ in reader)

            documents = []
            metadata_dict = {
                'source': self.file_path,
                'pages': 1,
                'num_rows': num_rows
            }

            metadata_dict.update(self.metadata)

            # Read the CSV file and create Document objects
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    row_content = ','.join(row)
                    document = Document(page_content=row_content, **metadata_dict)
                    documents.append(document)

            return documents
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Error decoding CSV file: {self.file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading the CSV file: {self.file_path}. Details: {e}")



