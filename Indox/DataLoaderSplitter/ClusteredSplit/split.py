from Indox.DataLoaderSplitter.ClusteredSplit.EmbedClusterSummarize import recursive_embed_cluster_summarize
from Indox.DataLoaderSplitter.ClusteredSplit.utils import get_all_texts, split_text, create_document
from Indox.DataLoaderSplitter.utils.clean import remove_stopwords_chunk
from typing import Optional, List, Tuple


def get_chunks(docs, embeddings, threshold, dim, chunk_size, overlap,
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
        leaf_chunks = split_text(texts, max_tokens=chunk_size, overlap=overlap)

        for i in range(len(leaf_chunks)):
            leaf_chunks[i] = leaf_chunks[i].replace("\n", " ")

        # Optionally remove stopwords from the chunks
        if remove_sword:
            leaf_chunks = remove_stopwords_chunk(leaf_chunks)

        results = recursive_embed_cluster_summarize(
            texts=leaf_chunks, embeddings=embeddings, dim=dim, threshold=threshold, level=1, n_levels=3,
            re_chunk=re_chunk, max_chunk=int(chunk_size / 2), remove_sword=remove_sword
        )
        all_chunks = get_all_texts(results=results, texts=leaf_chunks)

        print("End Chunking & Clustering process.")
        return all_chunks


    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise e


def ClusteredSplit(file_path: str,
                   embeddings,
                   re_chunk: bool = False,
                   remove_sword: bool = False,
                   chunk_size: Optional[int] = 100,
                   overlap: Optional[int] = 0,
                   threshold: float = 0.1,
                   dim: int = 10, ):
    all_chunks = get_chunks(docs=file_path,
                            chunk_size=chunk_size,
                            overlap=overlap,
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
