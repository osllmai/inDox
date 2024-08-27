class Sql:
    """
    Load a SQL file and extract its content.

    Parameters:
    - sql_path (str): The path to the SQL file to be loaded.

    Methods:
    - load(): Reads the SQL file and extracts its content.

    Returns:
    - str: The content of the SQL file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - UnicodeDecodeError: If there is an error decoding the SQL file.
    - RuntimeError: For any other errors encountered during SQL file processing.
    """

    def __init__(self, sql_path: str):
        self.sql_path = sql_path
        self.content = ""

    def load(self) -> str:
        try:
            with open(self.sql_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
            return self.content
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{self.sql_path}' does not exist.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("There was an error decoding the SQL file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the SQL file: {e}")


    def load_and_split(self, splitter, remove_stopwords=False):
        from indox.data_loaders.utils import load_and_process_input
        return load_and_process_input(loader=self.load, splitter=splitter, remove_stopwords=remove_stopwords)
