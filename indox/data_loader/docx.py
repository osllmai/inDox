
def Docx(file_path):
    """
    Load a DOCX file and extract its text and metadata, including estimated page numbers.

    Parameters:
    - file_path (str): The path to the DOCX file to be loaded.

    Returns:
    - List[Document]: A list containing `Document` objects with the text content and
      associated metadata, including estimated page numbers.

    Raises:
    - RuntimeError: If there is an error in loading the DOCX file.
    """
    import docx
    from indox.core.document_object import Document
    import os
    try:
        doc = docx.Document(file_path)
        paragraphs = doc.paragraphs

        paragraphs_per_page = 20
        num_pages = (len(paragraphs) + paragraphs_per_page - 1) // paragraphs_per_page

        # Extract text content
        documents = []
        for page in range(num_pages):
            start = page * paragraphs_per_page
            end = (page + 1) * paragraphs_per_page
            page_text = '\n'.join([p.text for p in paragraphs[start:end]])
            metadata_dict = {
                'source': os.path.abspath(file_path),
                'page': page
            }
            documents.append(Document(metadata=metadata_dict, page_content=page_text))

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading DOCX file: {e}")
