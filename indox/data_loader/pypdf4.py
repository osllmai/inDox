

def PyPdf4(file_path: str) -> List[Document]:
    """
    Load a PDF file and extract its text and metadata using PyPDF4.

    Parameters:
    - file_path (str): The path to the PDF file to be loaded.

    Returns:
    - List[Document]: A list of `Document` objects, each containing the text content of a page
      and the associated metadata.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - RuntimeError: If there is an error reading the PDF file or extracting text.
    """
    import PyPDF4
    from indox.core.document_object import Document
    from typing import List
    import os

    file_path = os.path.abspath(file_path)

    try:
        with open(file_path, 'rb') as f:
            try:
                reader = PyPDF4.PdfFileReader(f)
            except PyPDF4.utils.PdfReadError as e:
                raise RuntimeError(f"Error reading PDF file: {file_path}. Details: {e}")

            documents = []
            for i in range(reader.getNumPages()):
                page = reader.getPage(i)
                try:
                    text = page.extractText()
                except Exception as e:
                    raise RuntimeError(f"Error extracting text from page {i}. Details: {e}")

                metadata_dict = {
                    'source': file_path,
                    'page': i
                }

                document = Document(page_content=text, metadata=metadata_dict)
                documents.append(document)

            return documents
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while processing PDF file: {file_path}. Details: {e}")

