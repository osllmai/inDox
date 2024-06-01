from indox.data_loader_splitter.ClusteredSplit.EmbedClusterSummarize import recursive_embed_cluster_summarize
from indox.data_loader_splitter.ClusteredSplit.utils import get_all_texts, split_text, create_document
from indox.data_loader_splitter.utils.clean import remove_stopwords_chunk
from typing import Optional, List, Tuple


def get_chunks(docs, embeddings, threshold, dim, chunk_size, overlap,
               re_chunk, remove_sword, cluster_prompt, use_openai_summary, max_len_summary, min_len_summary):
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
        if cluster_prompt:
            texts = docs
        else:
            texts = create_document(docs)

        leaf_chunks = split_text(texts, max_tokens=chunk_size, overlap=overlap)

        for i in range(len(leaf_chunks)):
            leaf_chunks[i] = leaf_chunks[i].replace("\n", " ")

        # Optionally remove stopwords from the chunks
        if remove_sword:
            leaf_chunks = remove_stopwords_chunk(leaf_chunks)

        results = recursive_embed_cluster_summarize(
            texts=leaf_chunks, embeddings=embeddings, dim=dim, threshold=threshold, level=1, n_levels=3,
            re_chunk=re_chunk, max_chunk=int(chunk_size / 2), remove_sword=remove_sword,
            use_openai_summary=use_openai_summary, max_len_summary=max_len_summary, min_len_summary=min_len_summary
        )
        all_chunks = get_all_texts(results=results, texts=leaf_chunks)

        print("End Chunking & Clustering process.")
        return all_chunks


    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise e


# def ClusteredSplit(file_path: str,
#                    embeddings,
#                    re_chunk: bool = False,
#                    remove_sword: bool = False,
#                    chunk_size: Optional[int] = 100,
#                    overlap: Optional[int] = 0,
#                    threshold: float = 0.1,
#                    dim: int = 10):
#     all_chunks = get_chunks(docs=file_path,
#                             chunk_size=chunk_size,
#                             overlap=overlap,
#                             re_chunk=re_chunk,
#                             remove_sword=remove_sword,
#                             embeddings=embeddings,
#                             threshold=threshold,
#                             dim=dim,
#                             cluster_prompt=cluster_prompt)
#
#     return all_chunks

class ClusteredSplit:
    def __init__(self, file_path: str, embeddings, re_chunk: bool = False, remove_sword: bool = False,
                 chunk_size: Optional[int] = 100, overlap: Optional[int] = 0, threshold: float = 0.1, dim: int = 10,
                 use_openai_summary=False, max_len_summary=100, min_len_summary=30):
        self.file_path = file_path
        self.embeddings = embeddings
        self.re_chunk = re_chunk
        self.remove_sword = remove_sword
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.threshold = threshold
        self.dim = dim
        self.cluster_prompt = False
        self.use_openai_summary = use_openai_summary
        self.max_len_summary = max_len_summary
        self.min_len_summary = min_len_summary
    def get_all_docs(self):
        return get_chunks(docs=self.file_path,
                          chunk_size=self.chunk_size,
                          overlap=self.overlap,
                          re_chunk=self.re_chunk,
                          remove_sword=self.remove_sword,
                          embeddings=self.embeddings,
                          threshold=self.threshold,
                          dim=self.dim,
                          cluster_prompt=self.cluster_prompt,
                          use_openai_summary=self.use_openai_summary,
                          max_len_summary=self.max_len_summary,
                          min_len_summary=self.min_len_summary
                          )
