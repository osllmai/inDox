import time


# def get_metrics(inputs):
#     """
#     prints Precision, Recall and F1 obtained from BertScore
#     """
#     # mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
#     mP, mR, mF1, K = metrics(inputs)
#     print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
#     # print("\n\nUni Eval Sores")
#     # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]

def show_indox_logo():
    logo = """
            ██  ███    ██  ██████   ██████  ██       ██
            ██  ████   ██  ██   ██ ██    ██   ██  ██
            ██  ██ ██  ██  ██   ██ ██    ██     ██
            ██  ██  ██ ██  ██   ██ ██    ██   ██   ██
            ██  ██  █████  ██████   ██████  ██       ██
            """
    return print(logo)


def search_duckduckgo(query, max_retries=5, delay=2):
    from duckduckgo_search import DDGS
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
