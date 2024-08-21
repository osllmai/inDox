
def Rtf(file_path: str) -> List[Document]:
    """
    Load an RTF file and extract its text and metadata.

    Parameters:
    - file_path (str): The path to the RTF file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects containing the text content of the RTF file
      and associated metadata.

    Raises:
    - RuntimeError: If there is an error in loading the RTF file.

    Notes:
    - Metadata includes only 'source' and page number.
    """
    import pyth.plugins.rtf15.reader as rtf_reader
    import pyth.plugins.plaintext.writer as plaintext_writer
    from indox.core.document_object import Document
    from typing import List
    import os
    try:
        with open(file_path, 'rb') as f:
            doc = rtf_reader.read(f)
            text = plaintext_writer.write(doc).getvalue()

            metadata_dict = {
                'source': os.path.abspath(file_path),
                'page': 1
            }

            return [Document(page_content=text, metadata=metadata_dict)]
    except Exception as e:
        raise RuntimeError(f"Error loading RTF file: {e}")
