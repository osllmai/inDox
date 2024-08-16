# xml_loader.py
from indox.core.document_object import Document
import os
import time
import xml.etree.ElementTree as ET


def Xml(file_path):
    """
    Load an XML file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the XML file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of the XML file
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the XML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            text = ET.tostring(root, encoding='unicode', method='xml')

        # Extract metadata
        metadata_dict = {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'num_characters': len(text),
            'num_words': len(text.split()),
            'num_lines': text.count('\n') + 1,
            'last_modified': time.ctime(os.path.getmtime(file_path)),
            'file_size': os.path.getsize(file_path),
            'root_tag': root.tag,
            'num_elements': len(root.findall('.//')),  # Count all elements in the XML
        }

        # Create a Document object with the entire XML content
        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading XML file: {e}")
