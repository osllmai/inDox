import pandas as pd


class Excel:
    def __init__(self, file_path: str):
        """
        Initializes the ExcelLoader with the file path.

        Parameters:
            file_path (str): The path to the Excel file.
        """
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """
        Loads the Excel file and returns it as a DataFrame.

        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        try:
            df = pd.read_excel(self.file_path)
            return df
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return pd.DataFrame()
