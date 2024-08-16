from scipy.io import loadmat
from indox.core.document_object import Document

def Scipy(file_path):
    """
    Load a MATLAB .mat file and extract its contents as metadata.

    Parameters
    ----------
    file_path : str
        Path to the .mat file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the data variables
        from the .mat file as metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the .mat file.
    """
    try:
        mat_data = loadmat(file_path)

        # Remove MATLAB-specific metadata
        mat_data.pop('__header__', None)
        mat_data.pop('__version__', None)
        mat_data.pop('__globals__', None)

        documents = []

        for var_name, var_data in mat_data.items():
            document = Document(page_content=str(var_data), variable_name=var_name)
            documents.append(document)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading MATLAB file: {e}")
