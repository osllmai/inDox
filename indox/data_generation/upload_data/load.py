import os
import pandas as pd
from .csv import CSV
from .excel import Excel

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV or Excel file based on the file extension and return as a DataFrame."""
    _, ext = os.path.splitext(file_path)

    if ext.lower() == '.csv':
        csv_loader = CSV(file_path)
        documents = csv_loader.load()
    elif ext.lower() in ['.xls', '.xlsx']:
        excel_loader = Excel(file_path)
        all_documents = []
        sheets = excel_loader.load()
        for documents in sheets.values():
            all_documents.extend(documents)
        documents = all_documents
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    data = []
    for document in documents:
        data.append(document.page_content)

    return pd.DataFrame(data)
