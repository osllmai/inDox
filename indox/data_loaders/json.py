import json

class Json:
    """
    Load a JSON file and extract its content.

    Parameters:
    - json_path (str): The path to the JSON file to be loaded.

    Methods:
    - load(): Reads the JSON file and extracts its content.

    Returns:
    - dict: The content of the JSON file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - json.JSONDecodeError: If there is an error decoding the JSON file.
    - RuntimeError: For any other errors encountered during JSON file processing.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.content = {}

    def load(self) -> dict:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.content = json.load(file)
            return self.content
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.json_path}' does not exist.")
        except json.JSONDecodeError:
            raise json.JSONDecodeError("There was an error decoding the JSON file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the JSON file: {e}")

    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)

