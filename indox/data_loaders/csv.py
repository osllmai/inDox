from typing import List, Dict
from indox.vector_stores.utils import filter_complex_metadata
from indox.core import Document
import csv
import os


class CSV:
    """
    Load a CSV file and extract its data as Document objects.

    Parameters:
    - csv_path (str): The path to the CSV file to be loaded.

    Methods:
    - load(): Reads the CSV file, extracts data from each row, and creates a list of `Document` objects.

    Returns:
    - List[Document]: A list containing `Document` objects with the data of each row and associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: For any other errors encountered during CSV file processing.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data: List[Document] = []
        self.metadata = {}

    def load(self) -> List[Document]:
        try:
            with open(self.csv_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = [row for row in reader]

            self.metadata = {
                attr: getattr(os.stat(self.csv_path), attr)
                for attr in dir(os.stat(self.csv_path))
                if not attr.startswith('_') and not callable(getattr(os.stat(self.csv_path), attr))
            }

            filtered_csv_reader = filter_complex_metadata([self])[0]
            self.metadata = filtered_csv_reader.metadata

            self.data = [Document(page_content=row, **self.metadata) for row in rows]

            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.csv_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the CSV file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
