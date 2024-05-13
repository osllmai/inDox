# from .utils import create_document, read_config, create_documents_unstructured
# from .clean import remove_stopwords
from ..utils import read_config, create_documents_unstructured
import tiktoken
import re
from langchain_core.documents import Document
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from typing import List, Tuple, Optional, Any, Dict
# from .cluster.EmbedClusterSummarize import recursive_embed_cluster_summarize
# from .clean import remove_stopwords_chunk, remove_stopwords
from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores.utils import filter_complex_metadata


#
# def initialize_tokenizer(tokenizer_name: str):
#     """
#     Initialize a tokenizer based on the name provided.
#
#     Args:
#         tokenizer_name (str): Name of the tokenizer (e.g., "bert" or "openai").
#
#     Returns:
#         tokenizer: The initialized tokenizer object.
#     """
#     if tokenizer_name == "bert":
#         return Tokenizer.from_pretrained("bert-base-uncased")
#     elif tokenizer_name == "openai":
#         return tiktoken.get_encoding("cl100k_base")
#     else:
#         raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
#
#
# def initialize_splitter(splitter_name: str, tokenizer, max_tokens: int):
#     """
#     Initialize a text splitter based on the splitter name and tokenizer.
#
#     Args:
#         splitter_name (str): Name of the splitter ("semantic-text-splitter" or "raptor-text-splitter").
#         tokenizer: The tokenizer object.
#         max_tokens (int): Maximum number of tokens per chunk.
#
#     Returns:
#         splitter: The initialized text splitter object.
#     """
#     if splitter_name == "semantic-text-splitter":
#         if isinstance(tokenizer, tiktoken.Encoding):
#             return TextSplitter.from_tiktoken_model("gpt-3.5-turbo", max_tokens)
#         elif isinstance(tokenizer, Tokenizer):
#             return TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)
#         else:
#             raise ValueError("Unsupported tokenizer for semantic-text-splitter.")
#     else:
#         raise ValueError(f"Invalid splitter specified: {splitter_name}")
#




def get_chunks_unstructured(file_path, content_type, chunk_size: Optional[int] = 500, remove_sword=False):
    """
    Extract chunks from an unstructured document file using an unstructured data processing library.

    Parameters:
    - file_path (str): The path to the file containing unstructured data.
    - content_type (str): The type of content (e.g., "pdf", "html", etc.) to process.
    - chunk_size (int, optional): The maximum size (in characters) for each chunk. Defaults to 500.

    Returns:
    - list: A list of `Document` objects, each containing a portion of the original content with relevant metadata.

    Raises:
    - Exception: Any errors that occur during document processing or chunking.

    Notes:
    - The function uses a title-based chunking method to segment the unstructured data into logical parts.
    - Metadata is cleaned and filtered to ensure proper structure before being added to the `Document` objects.
    - The `filter_complex_metadata` function is used to simplify and sanitize metadata attributes.

    """
    try:
        print("Starting processing...")

        # Create initial document elements using the unstructured library
        elements = create_documents_unstructured(file_path, content_type=content_type)

        # Split elements based on the title and the specified max characters per chunk
        elements = chunk_by_title(elements, max_characters=chunk_size)

        documents = []

        # Convert each element into a `Document` object with relevant metadata
        for element in elements:
            metadata = element.metadata.to_dict()
            del metadata["languages"]  # Remove unnecessary metadata field

            for key, value in metadata.items():
                if isinstance(value, list):
                    value = str(value)
                metadata[key] = value

            if remove_sword == True:
                element.text = remove_stopwords(element.text)

            documents.append(Document(page_content=element.text, metadata=metadata))

        # Filter and sanitize complex metadata
        documents = filter_complex_metadata(documents=documents)

        print("End Chunking process.")
        return documents

    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise


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
