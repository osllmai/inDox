import pandas as pd


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


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)
