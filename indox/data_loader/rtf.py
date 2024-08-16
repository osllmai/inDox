import pyth.plugins.rtf15.reader as rtf_reader
import pyth.plugins.plaintext.writer as plaintext_writer
from indox.core.document_object import Document

def Rtf(file_path):
    """
    Load an RTF file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the RTF file to be loaded.

    Returns
    -------
    Document
        A `Document` object containing the text content of the RTF file.

    Raises
    ------
    RuntimeError
        If there is an error in loading the RTF file.
    """
    try:
        with open(file_path, 'rb') as f:
            doc = rtf_reader.read(f)
            text = plaintext_writer.write(doc).getvalue()
            return Document(page_content=text)
    except Exception as e:
        raise RuntimeError(f"Error loading RTF file: {e}")
