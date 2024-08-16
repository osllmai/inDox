import pickle
from indox.core.document_object import Document

def Pickle(file_path):
    """
    Load a PKL (Pickle) file and extract its content.

    Parameters
    ----------
    file_path : str
        Path to the PKL file to be loaded.

    Returns
    -------
    Document
        A `Document` object containing the unpickled content and associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the PKL file.
    """
    try:
        with open(file_path, 'rb') as f:
            content = pickle.load(f)

        # Create a Document object with metadata
        document = Document(page_content=str(content), file_type='pkl')

        return document
    except Exception as e:
        raise RuntimeError(f"Error loading PKL file: {e}")
