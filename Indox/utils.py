import PyPDF2
import pandas as pd
import yaml
import os

from .metrics.metrics import metrics

CONFIG_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def create_document(file_path):
    if file_path.lower().endswith(".pdf"):
        text = ""
        reader = PyPDF2.PdfReader(file_path)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r") as file:
            text = file.read()
    else:
        print(
            "Error: Unsupported document format. Please provide a string path to a PDF file or text."
        )

    return text


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def read_config() -> dict:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(current_directory, "config.yaml")
    with open(file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise RuntimeError("Can't open the config file.")


def construct_postgres_connection_string() -> str:
    config = read_config()
    conn_string = config["postgres"]["conn_string"]
    return conn_string


def reconfig(config: dict):
    """
        Edit a YAML file based on the provided dictionary.

        Args:
        - data_dict (dict): The dictionary containing the data to be written to the YAML file.
        - file_path (str): The file path of the YAML file to be edited.

        Returns:
        - None
    """

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_ = os.path.join(current_directory, "config.yaml")
    with open(file_, 'r') as file:
        existing_data = yaml.safe_load(file)

    existing_data.update(config)

    with open(file_, 'w') as file:
        yaml.dump(existing_data, file)


def get_user_input():
    response = input(
        "Would you like to add a clustering and summarization layer? This may double your token usage. Please select "
        "'y' for yes or 'n' for no: ")
    if response.lower() in ['y', 'n']:
        return response
    else:
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")


def get_metrics(inputs):
    """
    prints Precision, Recall and F1 obtained from BertScore
    """
    # mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
    mP, mR, mF1, K = metrics(inputs)
    print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
    # print("\n\nUni Eval Sores")
    # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]
