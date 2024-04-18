from .Embedding import embed_cluster_texts
from .utils import (
    create_document,
    fmt_txt,
    split_text,
    construct_postgres_connection_string,
)
from .Summary import summarize
from typing import List, Tuple, Optional, Any, Dict
import pandas as pd
from .QAModels import GPT3TurboQAModel
from .vectorstore import get_vector_store

def embed_cluster_summarize_texts(
    texts: List[str], embeddings, level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """
    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts, embeddings)

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    expanded_list = [
        {"text": row["text"], "embd": row["embd"], "cluster": cluster}
        for index, row in df_clusters.iterrows()
        for cluster in row["cluster"]
    ]
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()
    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarize the texts in each cluster
    summaries = []
    for cluster in all_clusters:
        cluster_texts = expanded_df[expanded_df["cluster"] == cluster]["text"].tolist()
        # Assuming `summarize` is a function that takes a list of texts and returns a summary
        summary = summarize(
            cluster_texts
        )  # Need to define or adjust this function accordingly
        summaries.append(summary)

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def get_all_texts(results, texts):
    all_texts = texts.copy()
    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)
    return all_texts


def recursive_embed_cluster_summarize(
    texts: List[str], embeddings, level: int = 1, n_levels: int = 3
):
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level using a specified embeddings object.

    Parameters:
    - texts: List[str], texts to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, embeddings, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, embeddings, level + 1, n_levels
        )
        results.update(next_level_results)

    return results


def get_chunks(docs, embeddings, max_tokens: Optional[int] = 500) -> List[str]:
    try:
        print("Starting processing...")
        texts = create_document(docs)
        leaf_chunks = split_text(
            texts, max_tokens=max_tokens
        )  # Pass max_tokens to split_text
        results = recursive_embed_cluster_summarize(
            texts=leaf_chunks, embeddings=embeddings, level=1, n_levels=3
        )
        all_chunks = get_all_texts(results=results, texts=leaf_chunks)
        return all_chunks
    except Exception as e:
        print(f"Failed at step with error: {e}")
        raise  # Re-raises the current exception to propagate the error up the call stack


class IndoxRetrievalAugmentation:
    def __init__(
        self,
        docs,
        embeddings,
        collection_name: str,
        qa_model: Optional[Any] = None,
        db: Optional[Any] = None,
        max_tokens: Optional[int] = 512,
    ):
        """
        Initialize the IndoxRetrievalAugmentation class with documents, embeddings object, an optional QA model, database connection, and maximum token count for text splitting.

        :param docs: List of documents to process
        :param embeddings: Embeddings object to be used for text processing
        :param qa_model: Optional pre-initialized QA model
        :param db: Optional pre-initialized database connection
        :param max_tokens: Optional maximum number of tokens for splitting texts
        """
        self.embeddings = embeddings
        self.db = db
        self.qa_model = qa_model if qa_model is not None else GPT3TurboQAModel()
        self.docs = docs
        self.max_tokens = max_tokens
        self.db = get_vector_store(collection_name=collection_name,
                                embeddings=embeddings)

    def get_all_chunks(self) -> List[str]:
        """
        Retrieve all chunks from the documents, using the specified maximum number of tokens if provided.
        """
        try:
            all_chunks = get_chunks(self.docs, self.embeddings, self.max_tokens)
            return all_chunks
        except Exception as e:
            print(f"Error while getting chunks: {e}")
            return []

    def store_in_vectorstore(self, all_chunks: List[str]) -> Any:
        """
        Store text chunks into a PostgreSQL database.
        """
        try:
            if self.db is not None:
                self.db.add_document(all_chunks)
            return self.db
        except Exception as e:
            print(f"Error while storing in PostgreSQL: {e}")
            return None

    def answer_question(
        self, query: str, top_k: int
    ) -> Tuple[str, List[float]]:
        """
        Answer a query using the QA model based on similar document chunks found in the database.
        """
        try:  
            context, scores = self.db.retrieve(query, top_k=top_k)
            answer = self.qa_model.answer_question(context=context, question=query)
            return answer, scores
        except Exception as e:
            print(f"Error while answering question: {e}")
            return "", []
