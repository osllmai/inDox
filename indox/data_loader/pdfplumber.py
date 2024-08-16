import pdfplumber
from indox.core.document_object import Document


def PdfPlumber(file_path):
    """
    Load a PDF file and extract its text and metadata using pdfplumber.

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
        with pdfplumber.open(file_path) as pdf:
            documents = []
            metadata_dict = pdf.metadata if pdf.metadata else {}

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()

                # Create a Document for each page
                document = Document(page_content=text, page_number=i + 1, **metadata_dict)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading PDF file: {e}")
