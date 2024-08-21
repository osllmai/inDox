from indox.core.document_object import Document
import os


def Sql(file_path):
    """
    Load an SQL file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the SQL file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of the SQL file
      and the associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the SQL file.

    Notes:
    - Metadata includes file details, content statistics, and the number of SQL statements.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract metadata
        metadata_dict = {
            'source': os.path.basename(file_path),
            'page': 1
        }

        # Create a Document object with the SQL content
        document = Document(page_content=text, **metadata_dict)

        return [document]
    except Exception as e:
        raise RuntimeError(f"Error loading SQL file: {e}")
