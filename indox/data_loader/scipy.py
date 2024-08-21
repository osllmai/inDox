

def Scipy(file_path: str) -> List[Document]:
    """
    Load a MATLAB .mat file and extract its contents as metadata.

    Parameters:
    - file_path (str): The path to the .mat file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the data variables
      from the .mat file as metadata.

    Raises:
    - RuntimeError: If there is an error in loading the .mat file.

    Notes:
    - MATLAB-specific metadata such as `__header__`, `__version__`, and `__globals__` are removed.
    - Metadata includes only 'source' and page number.
    """
    from scipy.io import loadmat
    from indox.core.document_object import Document
    from typing import List
    try:
        mat_data = loadmat(file_path)

        # Remove MATLAB-specific metadata
        mat_data.pop('__header__', None)
        mat_data.pop('__version__', None)
        mat_data.pop('__globals__', None)

        documents = []

        for i, (var_name, var_data) in enumerate(mat_data.items()):
            metadata_dict = {
                'source': file_path,
                'page': i
            }
            document = Document(page_content=str(var_data), metadata=metadata_dict)
            documents.append(document)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading MATLAB file: {e}")
