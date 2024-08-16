# md_loader.py
from indox.core.document_object import Document
import os
import time


def Md(file_path):
    """
    Load a Markdown file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the Markdown file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of the Markdown file
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the Markdown file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract metadata
        metadata_dict = {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'num_characters': len(text),
            'num_words': len(text.split()),
            'num_lines': text.count('\n') + 1,
            'last_modified': time.ctime(os.path.getmtime(file_path)),
            'file_size': os.path.getsize(file_path),
        }

        # Create a Document object with the entire Markdown content
        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading Markdown file: {e}")
