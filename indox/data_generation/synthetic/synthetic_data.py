import os
import pandas as pd
import json
import openpyxl
class SyntheticData:
    """Context manager for handling synthetic data generation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __enter__(self):
        # Set up synthetic data generation context
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory '{self.output_dir}' is ready.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exiting synthetic data context. Cleaning up if needed.")
        pass

    def save_data(self, filename: str, data):
        """Saves generated data to an Excel file."""
        df = data

        file_path = os.path.join(self.output_dir, filename)

        if not file_path.endswith('.xlsx'):
            file_path += '.xlsx'

        df.to_excel(file_path, index=False, engine='openpyxl')

        print(f"Data saved to '{file_path}' as Excel.")

    def save_data_to_excel(self, filename: str, data: list):
        """Saves generated data to an Excel file."""
        df = pd.DataFrame(data)
        excel_path = os.path.join(self.output_dir, filename)
        df.to_excel(excel_path, index=False)
        print(f"Data saved to Excel file '{excel_path}'.")

__all__ = ["SyntheticData"]
