import pandas as pd


class CSV:
    """Class to load CSV files into a pandas DataFrame."""

    def __init__(self, file_path: str):
        """Initialize with the file path."""
        self.file_path = file_path
        self.dataframe = None

    def load(self) -> pd.DataFrame:
        """Load the CSV file into a DataFrame."""
        try:
            self.dataframe = pd.read_csv(self.file_path)
            print(f"Successfully loaded {self.file_path}")
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
        except pd.errors.EmptyDataError:
            print("Error: No data found in the CSV file.")
        except pd.errors.ParserError:
            print("Error: Parsing the CSV file failed.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return self.dataframe
