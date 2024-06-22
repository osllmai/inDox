import logging
from indox.data_loader_splitter.ClusteredSplit.EmbedClusterSummarize import recursive_embed_cluster_summarize
from indox.data_loader_splitter.ClusteredSplit.cs_utils import get_all_texts, split_text, create_document
from indox.data_loader_splitter.utils.clean import remove_stopwords_chunk
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


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
      - If `re_chunk` is `True`, returns a tuple containing:
        - `all_chunks` (List[str]): A list of all document chunks (leaf and extra).

      - Otherwise, returns a list of leaf chunks without further clustering.

    Raises:
    - Exception: Any errors occurring during the chunking, clustering, or summarization process.

    Notes:
    - The function creates document chunks, optionally applies stopword removal, and clusters chunks based on the
      specified settings. The output varies depending on whether clustering is enabled or not.
    """
    try:
        logging.info("Starting processing for documents")

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

        logging.info("Completed chunking & clustering process")
        return all_chunks

    except Exception as e:
        logging.error("Failed at step with error: %s", e)
        raise e


class ClusteredSplit:
    def __init__(self, file_path: str, embeddings, re_chunk: bool = False, remove_sword: bool = False,
                 chunk_size: Optional[int] = 100, overlap: Optional[int] = 0, threshold: float = 0.1, dim: int = 10,
                 use_openai_summary: bool = False, max_len_summary: int = 100, min_len_summary: int = 30):
        """
        Initialize the ClusteredSplit class.

        Parameters:
        - file_path (str): The path to the file containing unstructured data.
        - embeddings: The embeddings to be used for processing the chunks.
        - re_chunk (bool, optional): Whether to re-chunk the document after initial chunking. Default is False.
        - remove_sword (bool, optional): Whether to remove stopwords from the text. Default is False.
        - chunk_size (int, optional): The maximum size (in characters) for each chunk. Default is 100.
        - overlap (int, optional): The number of characters to overlap between chunks. Default is 0.
        - threshold (float, optional): The similarity threshold for clustering. Default is 0.1.
        - dim (int, optional): The dimensionality of the embeddings. Default is 10.
        - use_openai_summary (bool, optional): Whether to use OpenAI summary for summarizing the chunks. Default is False.
        - max_len_summary (int, optional): The maximum length of the summary. Default is 100.
        - min_len_summary (int, optional): The minimum length of the summary. Default is 30.
        """
        try:
            logging.info("Initializing ClusteredSplit")
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
            logging.info("ClusteredSplit initialized successfully")
        except Exception as e:
            logging.error("Error initializing ClusteredSplit: %s", e)
            raise

    def load_and_chunk(self):
        """
        Split an unstructured document into chunks.

        Returns:
        - List[Document]: A list of `Document` objects, each containing a portion of the original content with relevant metadata.
        """
        try:
            logging.info("Getting all documents")
            docs = get_chunks(docs=self.file_path,
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
                              min_len_summary=self.min_len_summary)
            logging.info("Successfully obtained all documents")
            return docs
        except Exception as e:
            logging.error("Error in load_and_chunk: %s", e)
            raise

