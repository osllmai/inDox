# pptx_loader.py
from pptx import Presentation
from indox.core.document_object import Document
from typing import List, Dict


def Pptx(file_path: str) -> List[Document]:
    """
    Load a PowerPoint (.pptx) file and extract its text and metadata.

    Parameters
    ----------
    file_path : str
        Path to the PowerPoint file to be loaded.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each containing the text content of a slide
        and the associated metadata.

    Raises
    ------
    RuntimeError
        If there is an error in loading the PowerPoint file.
    """
    try:
        presentation = Presentation(file_path)
        documents = []

        # Create metadata
        metadata_dict = {
            'file_name': file_path,
            'num_slides': len(presentation.slides),
            # Add more metadata as needed
        }

        for i, slide in enumerate(presentation.slides):
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)

            # Create a Document for each slide
            document = Document(page_content='\n'.join(text), slide_number=i + 1, **metadata_dict)
            documents.append(document)

        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading PowerPoint file: {e}")
