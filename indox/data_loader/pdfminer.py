from pdfminer.high_level import extract_text, extract_pages
from pdfminer.pdfparser import PDFSyntaxError
from indox.core.document_object import Document


def PdfMiner(file_path):
    """
    Load a PDF file and extract its text and metadata using pdfminer.

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
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfinterp import resolve1

        documents = []

        with open(file_path, 'rb') as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            metadata_dict = resolve1(doc.info[0]) if doc.info else {}

            for i, page_layout in enumerate(extract_pages(file_path)):
                text = ''.join([element.get_text() for element in page_layout if hasattr(element, 'get_text')])
                document = Document(page_content=text, page_number=i + 1, **metadata_dict)
                documents.append(document)

        return documents
    except PDFSyntaxError as e:
        raise RuntimeError(f"Error loading PDF file (SyntaxError): {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading PDF file: {e}")
