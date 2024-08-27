class Txt:
    """
    Load a text file and extract its content.

    Parameters:
    - txt_path (str): The path to the text file to be loaded.

    Methods:
    - load(): Reads the text file and extracts its content.

    Returns:
    - str: The content of the text file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the text file.
    - RuntimeError: For any other errors encountered during text file processing.
    """

    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self.content = ""

    def load(self) -> str:
        try:
            with open(self.txt_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
            return self.content
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.txt_path}' does not exist.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("There was an error decoding the text file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the text file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)

