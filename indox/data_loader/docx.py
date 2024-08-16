# docx.py
from docx import Document as DocxDocument
from indox.core.document_object import Document

def Docx(file_path):
    """
    Load a DOCX file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the DOCX file to be loaded.

    Returns
    -------
    List[Document]
        A list containing a single `Document` object with the text content and
        associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the DOCX file.
    """
    try:
        doc = DocxDocument(file_path)

        # Extract metadata
        core_properties = doc.core_properties
        metadata_dict = {
            "author": core_properties.author,
            "title": core_properties.title,
            "subject": core_properties.subject,
            "keywords": core_properties.keywords,
            "last_modified_by": core_properties.last_modified_by,
            "created": core_properties.created,
            "modified": core_properties.modified,
            "category": core_properties.category,
            "comments": core_properties.comments,
            "content_status": core_properties.content_status,
            "identifier": core_properties.identifier,
            "language": core_properties.language,
            "version": core_properties.version
        }

        # Extract text content
        text_content = '\n'.join([p.text for p in doc.paragraphs])

        # Create a Document object
        document = Document(page_content=text_content, **metadata_dict)

        return [document]

    except Exception as e:
        raise RuntimeError(f"Error loading DOCX file: {e}")
