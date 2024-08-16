# txt_loader.py
from indox.core.document_object import Document
import os

def Txt(file_path):
    """
    Load a text file and return its content and metadata.

    Parameters
    ----------
    file_path : str
        Path to the text file to be loaded.

    Returns
    -------
    List[Document]
        A list containing a single `Document` object with the text content of the file
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the text file.
    """
    try:
        with open(file_path, 'r') as f:
            text = f.read()

        # Metadata extraction
        metadata_dict = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'file_path': file_path,
        }

        # Create a Document object with the text and metadata
        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading text file: {e}")

