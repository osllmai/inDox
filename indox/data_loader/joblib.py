
def Joblib(file_path: str) :
    """
    Load a PKL (Pickle) or Joblib file and extract its content.

    Parameters:
    - file_path (str): The path to the PKL or Joblib file to be loaded.

    Returns:
    - Document: A `Document` object containing the unpickled content and associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the file.

    Notes:
    - Metadata includes only 'source' and page number.
    """
    import joblib
    from indox.core.document_object import Document
    from typing import List
    import os

    try:
        content = joblib.load(file_path)

        metadata_dict = {
            'source': os.path.abspath(file_path),
            'page': 1
        }
        document = Document(page_content=str(content), metadata=metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")


