# import PyPDF2
# import pandas as pd
import yaml
import os
# import importlib
# from .metrics.metrics import metrics
# from unstructured.partition.pdf import partition_pdf
import latex2markdown

CONFIG_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


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


# def get_metrics(inputs):
#     """
#     prints Precision, Recall and F1 obtained from BertScore
#     """
#     # mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
#     mP, mR, mF1, K = metrics(inputs)
#     print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
#     # print("\n\nUni Eval Sores")
#     # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]


def convert_latex_to_md(latex_path):
    """Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file.

    Returns:
        str: The converted Markdown content, or None if there's an error.
    """
    try:
        with open(latex_path, 'r') as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        print(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def update_config(config):
    return reconfig(config)

def clear_log_file(file_path):
    with open(file_path, 'w') as file:
        pass