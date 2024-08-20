from indox.core.document_object import Document
import os
import time

def Md(file_path: str):
    """
    Load a Markdown file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the Markdown file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of the Markdown file
      and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the Markdown file.

    Notes:
    - Metadata includes the file path, a fixed page number  1.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        metadata_dict = {
            'source': os.path.abspath(file_path),
            'page': 1,
        }

        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading Markdown file: {e}")
