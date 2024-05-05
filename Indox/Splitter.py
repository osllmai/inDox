from .utils import create_document, read_config, create_documents_unstructured
import tiktoken
import re
from langchain_core.documents import Document
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from typing import List, Tuple, Optional, Any, Dict
from .cluster.EmbedClusterSummarize import recursive_embed_cluster_summarize
from .clean import remove_stopwords_chunk
from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores.utils import filter_complex_metadata


def split_text(text: str, max_tokens: int, overlap: int = 0) -> list:
    """
    Split the input text into chunks of approximately equal token counts, based on the specified maximum token count
    and overlap. The splitting approach depends on the system's configured splitter.

    Args:
    - text (str): The input text to be split into chunks.
    - max_tokens (int): The maximum number of tokens allowed in each chunk.
    - overlap (int, optional): The number of tokens to overlap between adjacent chunks. Defaults to 0.

    Returns:
    - list: A list of chunks, each containing a portion of the original text.

    Splitting logic:
    - If the "semantic-text-splitter" is configured, use the Hugging Face tokenizer with the specified maximum tokens.
    - If the "raptor-text-splitter" is configured, split by sentences and further refine by sub-sentences if any exceed
      the maximum token count.

    Notes:
    - The function reads a `splitter` configuration to determine which splitting method to apply.
    - Sentences that exceed `max_tokens` are split using internal delimiters like `,`, `;`, or `:`.
    - The `overlap` parameter allows for repeating tokens between adjacent chunks to provide better context.
    """
    config = read_config()

    if config["splitter"] == "semantic-text-splitter":
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)
        chunks = splitter.chunks(text)
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
            # Skip empty sentences
            if not sentence.strip():
                continue

            # Split overly long sentences into smaller chunks
            if token_count > max_tokens:
                sub_sentences = re.split(r"[,;:]", sentence)
                sub_token_counts = [
                    len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in sub_sentences
                ]

                sub_chunk = []
                sub_length = 0

                for sub_sentence, sub_token_count in zip(sub_sentences, sub_token_counts):
                    if sub_length + sub_token_count > max_tokens:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(
                            sub_token_counts[max(0, len(sub_chunk) - overlap): len(sub_chunk)]
                        )

                    sub_chunk.append(sub_sentence)
                    sub_length += sub_token_count

                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))

            # Start a new chunk if adding the current sentence exceeds the max limit
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

        # Add the last remaining chunk
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


def get_chunks(docs, embeddings, do_clustering, chunk_size: Optional[int] = 500,
               re_chunk: bool = False, remove_sword: bool = False):
    """
    Extract chunks from the provided documents using an embedding function. Optionally, recursively cluster
    and summarize the chunks.

    Parameters:
    - docs (Any): The source documents to be chunked and processed.
    - embeddings (Any): The embeddings model to use for clustering and summarization.
    - do_clustering (bool): If `True`, apply recursive clustering and summarization.
    - chunk_size (int, optional): The maximum size (in tokens) for each chunk. Defaults to 500.
    - re_chunk (bool, optional): Whether to re-chunk the clustered data for smaller summarization. Defaults to `False`.
    - remove_sword (bool, optional): If `True`, remove stopwords from the chunks. Defaults to `False`.

    Returns:
    - Tuple or List:
      - If `do_clustering` is `True`, returns a tuple containing:
        - `all_chunks` (List[str]): A list of all document chunks (leaf and extra).
        - `input_tokens_all` (int): The total number of input tokens used.
        - `output_tokens_all` (int): The total number of output tokens received.
      - Otherwise, returns a list of leaf chunks without further clustering.

    Raises:
    - Exception: Any errors occurring during the chunking, clustering, or summarization process.

    Notes:
    - The function creates document chunks, optionally applies stopword removal, and clusters chunks based on the
      specified settings. The output varies depending on whether clustering is enabled or not.
    """
    try:
        print("Starting processing...")

        # Create initial document chunks
        texts = create_document(docs)
        leaf_chunks = split_text(texts, max_tokens=chunk_size)

        for i in range(len(leaf_chunks)):
            leaf_chunks[i] = leaf_chunks[i].replace("\n", " ")

        # Optionally remove stopwords from the chunks
        if remove_sword:
            leaf_chunks = remove_stopwords_chunk(leaf_chunks)

        # Apply clustering if enabled
        if do_clustering:
            results, input_tokens_all, output_tokens_all = recursive_embed_cluster_summarize(
                texts=leaf_chunks, embeddings=embeddings, level=1, n_levels=3,
                re_chunk=re_chunk, max_chunk=int(chunk_size / 2), remove_sword=remove_sword
            )
            all_chunks = get_all_texts(results=results, texts=leaf_chunks)

            print(f"Create {len(all_chunks)} chunks: {len(leaf_chunks)} leaf chunks plus "
                  f"{int(len(all_chunks) - len(leaf_chunks))} extra chunks.")
            print("End Chunking & Clustering process.")
            return all_chunks, input_tokens_all, output_tokens_all
        else:
            all_chunks = leaf_chunks
            print(f"Create {len(all_chunks)} chunks.")
            print("End Chunking process.")
            return all_chunks

    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise


def get_chunks_unstructured(file_path, content_type, chunk_size: Optional[int] = 500):
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
