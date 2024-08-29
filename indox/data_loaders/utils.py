from indox.core import Document
from loguru import logger
import sys

from indox.core import Document

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


def load_and_process_input(loader, splitter, remove_stopwords=False):
    try:
        inputs = loader()

        if isinstance(inputs, str):
            # If inputs is a string, remove stopwords if requested
            if remove_stopwords:
                from indox.data_loader_splitter.utils.clean import remove_stopwords
                inputs = remove_stopwords(inputs)
            # Split the text
            chunks = splitter.split_text(inputs)

        elif isinstance(inputs, list) and all(isinstance(doc, Document) for doc in inputs):
            # If inputs is a list of Document objects
            text = ""
            for doc in inputs:
                text += doc.page_content
            # Remove stopwords if requested
            if remove_stopwords:
                from indox.data_loader_splitter.utils.clean import remove_stopwords
                text = remove_stopwords(text)
            # Split the concatenated text
            chunks = splitter.split_text(text)

        else:
            raise ValueError("Unsupported input type. Expected string or list of Document objects.")

        return chunks

    except Exception as e:
        raise RuntimeError(f"Error processing input: {e}")


def convert_latex_to_md(latex_path):
    """Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file.

    Returns:
        str: The converted Markdown content, or None if there's an error.
    """
    import latex2markdown
    try:
        with open(latex_path, 'r') as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        logger.info(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
