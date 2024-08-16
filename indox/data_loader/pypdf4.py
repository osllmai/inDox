import PyPDF4
from indox.core.document_object import Document


def PyPdf4(file_path):
    """
    Load a PDF file and extract its text and metadata using PyPDF4.

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
            reader = PyPDF4.PdfFileReader(f)
            documents = []
            metadata = reader.getDocumentInfo()

            # Convert metadata to a dictionary
            metadata_dict = {k[1:]: v for k, v in metadata.items()} if metadata else {}

            for i in range(reader.getNumPages()):
                page = reader.getPage(i)
                text = page.extractText()

                # Create a Document for each page
                document = Document(page_content=text, page_number=i + 1, **metadata_dict)
                documents.append(document)

            return documents
    except Exception as e:
        raise RuntimeError(f"Error loading PDF file: {e}")
