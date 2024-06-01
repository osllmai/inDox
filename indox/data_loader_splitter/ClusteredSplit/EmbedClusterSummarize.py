from typing import List
import pandas as pd
from .Embed import embed_cluster_texts
from .Summary import summarize
from .utils import rechunk
from ..utils.clean import remove_stopwords_chunk
import numpy as np


#
# def embed_cluster_summarize_texts(
#         texts: List[str], embeddings, dim, threshold, level: int, re_chunk=False, max_chunk: int = 100):
#     """
#     Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
#     clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
#     the content within each cluster.
#
#     Parameters:
#     - texts: A list of text documents to be processed.
#     - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
#     - level: An integer parameter that could define the depth or detail of processing.
#
#     Returns:
#     - Tuple containing two DataFrames:
#       1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
#       2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
#          and the cluster identifiers.
#     """
#     input_tokens_all = 0
#     output_tokens_all = 0
#     # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
#     df_clusters = embed_cluster_texts(texts, embeddings, dim, threshold)
#
#     # Expand DataFrame entries to document-cluster pairings for straightforward processing
#     expanded_list = [
#         {"text": row["text"], "embd": row["embd"], "cluster": cluster}
#         for index, row in df_clusters.iterrows()
#         for cluster in row["cluster"]
#     ]
#     expanded_df = pd.DataFrame(expanded_list)
#
#     # Retrieve unique cluster identifiers for processing
#     all_clusters = expanded_df["cluster"].unique()
#     print(f"--Generated {len(all_clusters)} clusters--")
#
#     # Summarize the texts in each cluster
#     summaries = []
#     for cluster in all_clusters:
#         cluster_texts = expanded_df[expanded_df["cluster"] == cluster]["text"].tolist()
#         summary, input_tokens, output_tokens = summarize(
#             cluster_texts
#         )
#         summaries.append(summary)
#         input_tokens_all += input_tokens
#         output_tokens_all += output_tokens
#     # Create a DataFrame to store summaries with their corresponding cluster and level
#     df_summary = pd.DataFrame(
#         {
#             "summaries": summaries,
#             "level": [level] * len(summaries),
#             "cluster": list(all_clusters),
#         }
#     )
#
#     if re_chunk:
#         df_summary = rechunk(df_summary=df_summary, max_chunk=max_chunk)
#
#     return df_clusters, df_summary, input_tokens_all, output_tokens_all
#
#
# def recursive_embed_cluster_summarize(texts: List[str], embeddings, dim, threshold, max_chunk: int = 100,
#                                       level: int = 1,
#                                       n_levels: int = 3,
#                                       re_chunk=False, remove_sword=False):
#     """
#     Recursively embeds, clusters, and summarizes texts up to a specified level or until
#     the number of unique clusters becomes 1, storing the results at each level using a specified embeddings object.
#
#     Parameters:
#     - texts: List[str], texts to be processed.
#     - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
#     - level: int, current recursion level (starts at 1).
#     - n_levels: int, maximum depth of recursion.
#
#     """
#     if remove_sword:
#         texts = remove_stopwords_chunk(texts)
#
#     results = {}
#
#     # Perform embedding, clustering, and summarization for the current level
#     df_clusters, df_summary, input_tokens, output_tokens = \
#         embed_cluster_summarize_texts(texts, embeddings, dim, threshold,
#                                       level, re_chunk=re_chunk,
#                                       max_chunk=max_chunk)
#     input_tokens_all = input_tokens
#     output_tokens_all = output_tokens
#     # Store the results of the current level
#     results[level] = (df_clusters, df_summary)
#
#     # Determine if further recursion is possible and meaningful
#     unique_clusters = df_summary["cluster"].nunique()
#     if level < n_levels and unique_clusters > 1:
#         new_texts = df_summary["summaries"].tolist()
#         next_level_results, input_tokens, output_tokens = recursive_embed_cluster_summarize(
#             new_texts, embeddings, level + 1, n_levels
#         )
#         input_tokens_all += input_tokens
#         output_tokens_all += output_tokens
#         results.update(next_level_results)
#
#     return results, input_tokens_all, output_tokens_all
def embed_cluster_summarize_texts(
        texts: List[str], embeddings, dim: int, threshold: float, level: int,
        use_openai_summary: bool, max_len_summary: int, min_len_summary: int,
        re_chunk: bool = False,
        max_chunk: int = 100):
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
    - dim: int, the dimension of the embeddings.
    - threshold: float, the clustering threshold.
    - level: int, an integer parameter that could define the depth or detail of processing.
    - re_chunk: bool, whether to re-chunk the summaries.
    - max_chunk: int, the maximum size of a chunk.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts, embeddings, dim, threshold)

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    expanded_list = [
        {"text": row["text"], "embd": row["embd"], "cluster": row["cluster"]}
        for index, row in df_clusters.iterrows()
    ]
    for item in expanded_list:
        if isinstance(item["cluster"], np.ndarray):
            item["cluster"] = tuple(item["cluster"])
    # expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    # all_clusters = expanded_df["cluster"].unique()
    # print(f"--Generated {len(all_clusters)} clusters--")

    # Create DataFrame
    expanded_df = pd.DataFrame(expanded_list)
    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()
    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarize the texts in each cluster (placeholder for actual summarization logic)
    # for cluster in all_clusters:
    #     cluster_texts = expanded_df[expanded_df["cluster"] == cluster]["text"].tolist()
    # Summarize the texts in each cluster
    summaries = []
    for cluster in all_clusters:
        cluster_texts = expanded_df[expanded_df["cluster"] == cluster]["text"].tolist()
        summary = summarize(
            cluster_texts, use_openai=use_openai_summary, max_len=max_len_summary, min_len=min_len_summary
        )
        summaries.append(summary)

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

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(texts: List[str], embeddings, dim: int, threshold: float,
                                      use_openai_summary: bool, max_len_summary: int, min_len_summary: int,
                                      max_chunk: int = 100,
                                      level: int = 1, n_levels: int = 3,
                                      re_chunk: bool = False, remove_sword: bool = False,
                                      ):
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level using a specified embeddings object.

    Parameters:
    - texts: List[str], texts to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.
    - dim: int, the dimension of the embeddings.
    - threshold: float, the clustering threshold.
    - max_chunk: int, the maximum size of a chunk.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.
    - re_chunk: bool, whether to re-chunk the summaries.
    - remove_sword: bool, whether to remove stop words.

    Returns:
    - A tuple containing the results, input tokens, and output tokens.
    """
    if remove_sword:
        texts = remove_stopwords_chunk(texts)

    results = {}

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, embeddings, dim, threshold, level,
                                                            use_openai_summary=use_openai_summary,
                                                            max_len_summary=max_len_summary,
                                                            min_len_summary=min_len_summary,
                                                            re_chunk=re_chunk,
                                                            max_chunk=max_chunk)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, embeddings, dim, threshold, max_chunk, level + 1, n_levels, re_chunk, remove_sword
        )

        results.update(next_level_results)

    return results
