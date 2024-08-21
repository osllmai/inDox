from indox.core.document_object import Document
from typing import List
def Bs4(file_path: str) -> List[Document]:
    """
    Load an HTML file and extract its text and minimal metadata.

    Parameters:
    - file_path (str): The path to the HTML file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of the HTML file
      and the associated minimal metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the HTML file.
    - RuntimeError: For any other errors encountered during HTML processing.
    """
    from bs4 import BeautifulSoup

    import os
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Error decoding HTML file: {file_path}. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading the HTML file: {file_path}. Details: {e}")

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
        except Exception as e:
            raise RuntimeError(f"Error parsing HTML content: {file_path}. Details: {e}")

        metadata_dict = {
            'source': os.path.abspath(file_path),
            'page': 1,
        }

        try:
            document = Document(page_content=text, **metadata_dict)
        except Exception as e:
            raise RuntimeError(f"Error creating Document object: {file_path}. Details: {e}")

        return [document]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}. Details: {e}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f"Error decoding HTML file: {file_path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {file_path}. Details: {e}")
