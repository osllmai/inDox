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

def split_text(text: str, max_tokens, overlap: int = 0):
    """
    Splits the input text into chunks of approximately equal token counts, based on the specified maximum token count
    and overlap. The method of splitting depends on the configured splitter in the system.

    Args:
        text (str): The input text to be split into chunks.
        max_tokens (int): The maximum number of tokens allowed in each chunk.
        overlap (int, optional): The number of tokens to overlap between adjacent chunks. Defaults to 0.

    Returns:
        list: A list of chunks, each containing a portion of the original text.
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


def get_chunks(docs, embeddings, do_clustering, chunk_size: Optional[int] = 500,
               re_chunk: bool = False, remove_sword=False):
    """
    Extract chunks using an embedding function and recursively
    """
    try:
        print("Starting processing...")
        texts = create_document(docs)
        leaf_chunks = split_text(texts, max_tokens=chunk_size)
        for i in range(len(leaf_chunks)):
            leaf_chunks[i] = leaf_chunks[i].replace("\n", " ")

        if remove_sword == True:
            leaf_chunks = remove_stopwords_chunk(leaf_chunks)

        if do_clustering:
            results, input_tokens_all, output_tokens_all = recursive_embed_cluster_summarize(
                texts=leaf_chunks, embeddings=embeddings, level=1, n_levels=3,
                re_chunk=re_chunk, max_chunk=int(chunk_size / 2), remove_sword=remove_sword
            )
            all_chunks = get_all_texts(results=results, texts=leaf_chunks)
            print(
                f"Create {len(all_chunks)} Chunks, {len(leaf_chunks)} leaf chunks plus {int(len(all_chunks) - len(leaf_chunks))} extra chunks")
            print("End Chunking & Clustering process")
            return all_chunks, input_tokens_all, output_tokens_all
        elif not do_clustering:
            all_chunks = leaf_chunks
            print(f"Create {len(all_chunks)} Chunks")
            print("End Chunking process")
            return all_chunks
    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise  # Re-raises the current exception to propagate the error up the call stack

def get_chunks_unstructured(file_path, content_type, chunk_size: Optional[int] = 500):
    """
    Extract chunks using unstructured library
    """
    try:
        print("Starting processing...")
        elements = create_documents_unstructured(file_path, content_type=content_type)
        elements = chunk_by_title(elements, max_characters=chunk_size)
        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            del metadata["languages"]
            for key, value in metadata.items():
                if isinstance(value, list):
                    value = str(value)
                metadata[key] = value

            documents.append(Document(page_content=element.text, metadata=metadata))
            documents = filter_complex_metadata(documents=documents)
        print("End Chunking process")
        return documents

    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise  # Re-raises the current exception to propagate the error up the call stack


def rechunk(df_summary, max_chunk):
    re_chunked = []
    for i, row in df_summary.iterrows():
        chunks = split_text(row['summaries'], max_chunk)
        re_chunked.append(chunks)
    df_summary['summaries'] = re_chunked
    df_summary = df_summary.explode('summaries')
    df_summary.reset_index(drop=True, inplace=True)
    return df_summary
