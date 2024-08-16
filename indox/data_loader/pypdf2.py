# pdf_loader.py
import PyPDF2
from indox.core.document_object import Document


def PyPdf2(file_path):
    """
    Load a PDF file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the PDF file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of a page
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the PDF file.
    """
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            documents = []
            metadata = reader.metadata

            # Convert metadata to a dictionary
            metadata_dict = {k[1:]: v for k, v in metadata.items()} if metadata else {}

            for i, page in enumerate(reader.pages):
                text = page.extract_text()

                # Create a Document for each page
                document = Document(page_content=text, page_number=i + 1, **metadata_dict)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading PDF file: {e}")
