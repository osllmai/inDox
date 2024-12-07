import nltk


def create_model_dict(name, score, metrics):
    """
    Creates a dictionary for a model with the given name, score, and metrics.

    Parameters:
    name (str): The name of the model.
    score (float): The score of the model.
    metrics (dict): A dictionary containing the metrics and their values.

    Returns:
    dict: A dictionary representing the model with its name, score, and metrics.
    """

    return {
        'name': name,
        'score': score,
        'metrics': metrics
    }


def nltk_download():
    from nltk import download

    download('punkt', quiet=True)
    download('stopwords', quiet=True)
    download('wordnet', quiet=True)
    download('averaged_perceptron_tagger', quiet=True)


