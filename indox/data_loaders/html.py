class Html:
    """
    Load an HTML file and extract its content.

    Parameters:
    - html_path (str): The path to the HTML file to be loaded.

    Methods:
    - load(): Reads the HTML file and extracts its content.

    Returns:
    - str: The content of the HTML file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the HTML file.
    - RuntimeError: For any other errors encountered during HTML file processing.
    """

    def __init__(self, html_path: str):
        self.html_path = html_path
        self.content = ""

    def load(self) -> str:
        try:
            with open(self.html_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
            return self.content
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.html_path}' does not exist.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("There was an error decoding the HTML file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the HTML file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
