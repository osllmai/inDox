from typing import List
import pandas as pd
from .Embed import embed_cluster_texts
from .Summary import summarize
from .utils import rechunk
from ..utils.clean import remove_stopwords_chunk


def embed_cluster_summarize_texts(
        texts: List[str], embeddings, dim, threshold, level: int, re_chunk=False, max_chunk: int = 100):
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
    input_tokens_all = 0
    output_tokens_all = 0
    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts, embeddings, dim, threshold)

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
        summary, input_tokens, output_tokens = summarize(
            cluster_texts
        )
        summaries.append(summary)
        input_tokens_all += input_tokens
        output_tokens_all += output_tokens
    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    if re_chunk:
        df_summary = rechunk(df_summary=df_summary, max_chunk=max_chunk)

    return df_clusters, df_summary, input_tokens_all, output_tokens_all


def recursive_embed_cluster_summarize(texts: List[str], embeddings, dim, threshold, max_chunk: int = 100,
                                      level: int = 1,
                                      n_levels: int = 3,
                                      re_chunk=False, remove_sword=False):
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level using a specified embeddings object.

    Parameters:
    - texts: List[str], texts to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    """
    if remove_sword:
        texts = remove_stopwords_chunk(texts)

    results = {}

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary, input_tokens, output_tokens = \
        embed_cluster_summarize_texts(texts, embeddings, dim, threshold,
                                      level, re_chunk=re_chunk,
                                      max_chunk=max_chunk)
    input_tokens_all = input_tokens
    output_tokens_all = output_tokens
    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results, input_tokens, output_tokens = recursive_embed_cluster_summarize(
            new_texts, embeddings, level + 1, n_levels
        )
        input_tokens_all += input_tokens
        output_tokens_all += output_tokens
        results.update(next_level_results)

    return results, input_tokens_all, output_tokens_all
