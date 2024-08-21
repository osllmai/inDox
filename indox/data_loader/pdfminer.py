
def PdfMiner(file_path: str) -> List[Document]:
    """
    Load a PDF file and extract its text and metadata using pdfminer.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of a page
      and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - PDFSyntaxError: If there is a syntax-related issue in the PDF file.
    - RuntimeError: For any other errors encountered during PDF processing.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.pdfparser import PDFSyntaxError
    from indox.core.document_object import Document
    from typing import List
    import os
    file_path = os.path.abspath(file_path)

    try:
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfinterp import resolve1

        documents = []
        try:
            with open(file_path, 'rb') as f:
                parser = PDFParser(f)
                doc = PDFDocument(parser)


                for i, page_layout in enumerate(extract_pages(file_path)):
                    text = ''.join([element.get_text() for element in page_layout if hasattr(element, 'get_text')])

                    metadata_dict = {
                        'source': file_path,
                        'page': i
                    }
                    document = Document(page_content=text, metadata=metadata_dict)
                    documents.append(document)

            return documents
        except PDFSyntaxError as e:
            raise PDFSyntaxError(f"Error loading PDF file (SyntaxError): {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while processing PDF file: {file_path}. Details: {e}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

