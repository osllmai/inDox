from indox.core.document_object import Document
from typing import List


class Scipy:
    """
    Load a MATLAB .mat file and extract its contents as metadata.

    Parameters:
    - file_path (str): The path to the .mat file to be loaded.

    Methods:
    - load_file(): Loads the .mat file and returns a list containing `Document` objects with
                   the data variables from the file as metadata.

    Raises:
    - RuntimeError: If there is an error in loading the .mat file.

    Notes:
    - MATLAB-specific metadata such as `__header__`, `__version__`, and `__globals__` are removed.
    - Metadata includes only 'source' and page number.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        from scipy.io import loadmat
        try:
            mat_data = loadmat(self.file_path)

            # Remove MATLAB-specific metadata
            mat_data.pop('__header__', None)
            mat_data.pop('__version__', None)
            mat_data.pop('__globals__', None)

            documents = []

            for i, (var_name, var_data) in enumerate(mat_data.items()):
                metadata_dict = {
                    'source': self.file_path,
                    'page': i
                }
                document = Document(page_content=str(var_data), metadata=metadata_dict)
                documents.append(document)

            return documents

        except Exception as e:
            raise RuntimeError(f"Error loading MATLAB file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loader.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
