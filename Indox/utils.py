import PyPDF2
import pandas as pd
import yaml
import os
import importlib
from .metrics.metrics import metrics
from unstructured.partition.pdf import partition_pdf
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




def get_metrics(inputs):
    """
    prints Precision, Recall and F1 obtained from BertScore
    """
    # mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
    mP, mR, mF1, K = metrics(inputs)
    print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
    # print("\n\nUni Eval Sores")
    # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]


def create_documents_unstructured(file_path: str, content_type: str):
    """
    Process unstructured documents and return the parsed elements based on the specified content type.

    Parameters:
    - file_path (str): The path to the document file to be processed.
    - content_type (str): The type of the document (e.g., "pdf", "html", "md", "tex") to determine the processing method.

    Returns:
    - list: A list of document elements extracted and partitioned from the specified file.
    - Exception: The exception object if an error occurs.

    Notes:
    - For PDFs, a high-resolution strategy is used, along with reference filtering.
    - For other types, the `partition` functions from the `unstructured` library are used.
    - LaTeX files are converted to Markdown before partitioning.

    Raises:
    - ImportError: If the appropriate partitioning module cannot be imported.
    - AttributeError: If the specific partitioning function is unavailable in the module.
    - Exception: Any other unexpected errors that occur during the document partitioning process.
    """
    elements = None
    latex_file = False

    try:
        if file_path.lower().endswith(".pdf") or content_type == "pdf":
            # Partition PDF with a high-resolution strategy
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                model_name="yolox"
            )

            # Remove "References" and header elements
            reference_title = [
                el for el in elements
                if el.text == "References" and el.category == "Title"
            ][0]
            references_id = reference_title.id
            elements = [el for el in elements if el.metadata.parent_id != references_id]
            elements = [el for el in elements if el.category != "Header"]

        else:
            if content_type == "tex":
                content_type = "md"
                latex_file = True

            # Import appropriate partition function from the `unstructured` library
            module_name = f"unstructured.partition.{content_type}"
            module = importlib.import_module(module_name)
            partition_function_name = f"partition_{content_type}"
            prt = getattr(module, partition_function_name)

            # Partition based on the file type
            if content_type == "html":
                elements = prt(url=file_path)
            elif content_type == "md" and not latex_file:
                elements = prt(filename=file_path)
            elif content_type == "text":
                elements = prt(filename=file_path)
            elif latex_file:
                md_text = convert_latex_to_md(latex_path=file_path)
                elements = prt(text=md_text)

        return elements

    except ImportError as ie:
        print(f"Module import error: {ie}")
        return ie
    except AttributeError as ae:
        print(f"Attribute error: {ae}")
        return ae
    except Exception as e:
        print(f"Unexpected error: {e}")
        return e


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
