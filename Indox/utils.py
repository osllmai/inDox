import PyPDF2
import pandas as pd
import yaml
import os
import importlib
from .metrics.metrics import metrics
from unstructured.partition.pdf import partition_pdf
import latex2markdown

CONFIG_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def create_document(file_path: str) -> str:
    """
    Extract the text content from a specified document file.

    Parameters:
    - file_path (str): The path to the document file to be processed. Supported formats are PDF and plain text.

    Returns:
    - str: The text content extracted from the document.

    Raises:
    - ValueError: If the file extension is not `.pdf` or `.txt`.
    - FileNotFoundError: If the specified file path does not exist.

    Notes:
    - Uses the `PyPDF2` library for PDF extraction and standard file I/O for plain text files.
    - Handles case-insensitive extensions.

    """
    # Check for valid file extensions and process accordingly
    if file_path.lower().endswith(".pdf"):
        text = ""
        try:
            reader = PyPDF2.PdfReader(file_path)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file: {e}")

    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, "r") as file:
                text = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading text file: {e}")

    else:
        raise ValueError("Unsupported document format. Please provide a PDF or plain text file.")

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


def get_user_input() -> str:
    """
    Prompt the user for input on whether to add a clustering and summarization layer.

    Returns:
    - str: The user's response, either 'y' for yes or 'n' for no.

    Raises:
    - ValueError: If the input is not 'y' or 'n'.

    Notes:
    - The function loops until the user provides a valid input, ensuring that only 'y' or 'n' are returned.
    """
    while True:
        response = input(
            "Would you like to add a clustering and summarization layer? This may double your token usage. "
            "Please select 'y' for yes or 'n' for no: "
        ).strip().lower()

        if response in ['y', 'n']:
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
