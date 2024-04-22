import tiktoken
import re
import PyPDF2
import pandas as pd
import yaml
import os
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

CONFIG_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def split_text(text: str, max_tokens, overlap: int = 0):
    config = read_config()
    if config["splitter"] == "semantic-text-splitter":
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)
        chunks = splitter.chunks(text, max_tokens)
        return chunks
    elif config["splitter"] == "raptor-text-splitter":
        tokenizer = tiktoken.get_encoding("cl100k_base")
        delimiters = [".", "!", "?", "\n"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        sentences = re.split(regex_pattern, text)
        # Calculate the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence, token_count in zip(sentences, n_tokens):
            # If the sentence is empty or consists only of whitespace, skip it
            if not sentence.strip():
                continue
            # If the sentence is too long, split it into smaller parts
            if token_count > max_tokens:
                sub_sentences = re.split(r"[,;:]", sentence)
                sub_token_counts = [
                    len(tokenizer.encode(" " + sub_sentence))
                    for sub_sentence in sub_sentences
                ]

                sub_chunk = []
                sub_length = 0

                for sub_sentence, sub_token_count in zip(sub_sentences, sub_token_counts):
                    if sub_length + sub_token_count > max_tokens:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(
                            sub_token_counts[
                            max(0, len(sub_chunk) - overlap): len(sub_chunk)
                            ]
                        )

                    sub_chunk.append(sub_sentence)
                    sub_length += sub_token_count

                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))

            # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
            elif current_length + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(
                    n_tokens[max(0, len(current_chunk) - overlap): len(current_chunk)]
                )
                current_chunk.append(sentence)
                current_length += token_count

            # Otherwise, add the sentence to the current chunk
            else:
                current_chunk.append(sentence)
                current_length += token_count

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


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
