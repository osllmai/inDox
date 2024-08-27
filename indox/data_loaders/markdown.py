class Markdown:
    """
    Load a Markdown file and extract its content.

    Parameters:
    - md_path (str): The path to the Markdown file to be loaded.

    Methods:
    - load(): Reads the Markdown file and extracts its content.

    Returns:
    - str: The content of the Markdown file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the Markdown file.
    - RuntimeError: For any other errors encountered during Markdown file processing.
    """

    def __init__(self, md_path: str):
        self.md_path = md_path
        self.content = ""

    def load(self) -> str:
        try:
            with open(self.md_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
            return self.content
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.md_path}' does not exist.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("There was an error decoding the Markdown file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the Markdown file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
