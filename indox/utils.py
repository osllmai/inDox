import latex2markdown
import time
from duckduckgo_search import DDGS

# def get_metrics(inputs):
#     """
#     prints Precision, Recall and F1 obtained from BertScore
#     """
#     # mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
#     mP, mR, mF1, K = metrics(inputs)
#     print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
#     # print("\n\nUni Eval Sores")
#     # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]


def convert_latex_to_md(latex_path):
    """Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file.

    Returns:
        str: The converted Markdown content, or None if there's an error.
    """
    try:
        with open(latex_path, 'r') as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        print(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None




def clear_log_file(file_path):
    with open(file_path, 'w') as file:
        pass


def search_duckduckgo(query, max_retries=5, delay=2):
    ddgs = DDGS()
    for attempt in range(max_retries):
        results = []
        try:
            result = ddgs.text(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                max_results=5
            )
            for res in result:
                results.append(res['body'])
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("Max retries reached. Exiting.")
    return None
