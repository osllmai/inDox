

def create_document(file_path: str) -> str:
    """
    Extract the text content from a specified document file.

    Parameters:
    - file_path (str): The path to the document file to be processed. Supported formats are PDF and plain text.

    Returns:
    - str: The text content extracted from the document.

    Raises:
    - ValueError: If the file extension is not `.pdf` or `.txt`.
    - FileNotFoundError: If the specified file path does not exist.

    Notes:
    - Uses the `PyPDF2` library for PDF extraction and standard file I/O for plain text files.
    - Handles case-insensitive extensions.

    """
    import PyPDF2

    # Check for valid file extensions and process accordingly
    if file_path.lower().endswith(".pdf"):
        text = ""
        try:
            reader = PyPDF2.PdfReader(file_path)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file: {e}")

    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, "r") as file:
                text = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading text file: {e}")

    else:
        raise ValueError("Unsupported document format. Please provide a PDF or plain text file.")

    return text
