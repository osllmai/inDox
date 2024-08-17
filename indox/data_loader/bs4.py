# html_loader.py
from bs4 import BeautifulSoup
from indox.core.document_object import Document
from typing import List, Dict
import os
import time


def Bs4(file_path: str) -> List[Document]:
    """
    Load an HTML file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the HTML file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of the HTML file
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the HTML file.
    """
    try:
        # Read HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()

        # Extract metadata
        metadata_dict = {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'num_characters': len(text),
            'num_words': len(text.split()),
            'num_paragraphs': len(soup.find_all('p')),
            'num_images': len(soup.find_all('img')),
            'num_links': len(soup.find_all('a')),
            'last_modified': time.ctime(os.path.getmtime(file_path)),
            'file_size': os.path.getsize(file_path),
        }

        # Create a Document object with the entire HTML content
        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading HTML file: {e}")
