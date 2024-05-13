import tiktoken

from Indox.splitter.spltiWithClustering.EmbedClusterSummarize import recursive_embed_cluster_summarize
from Indox.splitter.spltiWithClustering.utils import get_all_texts
from Indox.utils import create_document
from .clean import remove_stopwords_chunk, remove_stopwords
from typing import Optional, List, Tuple
import re


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
        if not sentence.strip():
            continue

            # Split long sentences into sub-chunks
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap): len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

            # Start a new chunk if adding this sentence exceeds the limit
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap): len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    else:
        raise ValueError("Invalid splitter specified in the configuration.")


def get_chunks(docs, embeddings, threshold, dim, chunk_size,
               re_chunk, remove_sword):
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

        results, input_tokens_all, output_tokens_all = recursive_embed_cluster_summarize(
            texts=leaf_chunks, embeddings=embeddings, dim=dim, threshold=threshold, level=1, n_levels=3,
            re_chunk=re_chunk, max_chunk=int(chunk_size / 2), remove_sword=remove_sword
        )
        all_chunks = get_all_texts(results=results, texts=leaf_chunks)

        print("End Chunking & Clustering process.")
        # return all_chunks, input_tokens_all, output_tokens_all
        return all_chunks


    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise e


def SplitWithClustering(file_path: str,
                        embeddings,
                        re_chunk: bool = False,
                        remove_sword: bool = False,
                        max_chunk_size: Optional[int] = 500,
                        threshold: float = 0.1,
                        dim: int = 10, ):
    all_chunks = get_chunks(docs=file_path,
                            chunk_size=max_chunk_size,
                            re_chunk=re_chunk,
                            remove_sword=remove_sword,
                            embeddings=embeddings,
                            threshold=threshold,
                            dim=dim)
    # encoding = tiktoken.get_encoding(encoding)
    # embedding_tokens = 0
    # for chunk in all_chunks:
    #     token_count = len(encoding.encode(chunk))
    #     embedding_tokens = embedding_tokens + token_count
    # return all_chunks, input_tokens_all, output_tokens_all
    return all_chunks
