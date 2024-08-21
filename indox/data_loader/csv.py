import csv
from indox.core.document_object import Document
from typing import List, Dict, Any
import os


def Csv(file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Load a CSV file and return its content as a list of `Document` objects.

    Parameters:
    - file_path (str): The path to the CSV file to be loaded.
    - metadata (dict, optional): Additional metadata to include in each `Document`. Default is None.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the CSV rows and associated metadata.


    Notes:
    - Metadata can be customized and will be included in each `Document` object along with the CSV row content.
    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # First, determine the number of rows in the CSV file
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                num_rows = sum(1 for _ in reader)  # Count the number of rows

            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                documents = []
                metadata_dict = {
                    'source': os.path.abspath(file_path),
                    'pages': 1
                }

                if metadata:
                    metadata_dict.update(metadata)

                for i, row in enumerate(reader):
                    row_content = ','.join(row)

                    # Create a Document for each row
                    document = Document(page_content=row_content, **metadata_dict)
                    documents.append(document)

                return documents
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Error decoding CSV file: {file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading the CSV file: {file_path}. Details: {e}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {file_path}. Details: {e}")
