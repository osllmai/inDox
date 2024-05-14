import PyPDF2
import pandas as pd
import tiktoken
import re


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


def split_text(text: str, max_tokens: int, overlap: int = 0):
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


def get_all_texts(results, texts):
    """
    Extracts text from summaries
    """
    all_texts = texts.copy()
    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)
    return all_texts


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


def rechunk(df_summary, max_chunk: int):
    """
    Re-chunk the summaries in the DataFrame into smaller chunks based on a specified maximum chunk size.

    Parameters:
    - df_summary (pandas.DataFrame): The DataFrame containing summary text to be re-chunked.
    - max_chunk (int): The maximum size (in tokens) allowed for each chunk.

    Returns:
    - pandas.DataFrame: A new DataFrame where each row contains one of the smaller chunks.

    Notes:
    - The function splits the summary text in each row into smaller chunks using the `split_text` function.
    - The `explode` method is used to expand the list of chunks into individual rows for each summary.
    """
    re_chunked = []

    # Split the summary text in each row into smaller chunks
    for _, row in df_summary.iterrows():
        chunks = split_text(row['summaries'], max_chunk)
        re_chunked.append(chunks)

    # Update the 'summaries' column with the new chunks
    df_summary['summaries'] = re_chunked

    # Explode the list into individual rows and reset the index
    df_summary = df_summary.explode('summaries')
    df_summary.reset_index(drop=True, inplace=True)

    return df_summary
