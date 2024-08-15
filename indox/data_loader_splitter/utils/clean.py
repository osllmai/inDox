# from typing import List
#
#
# def download():
#     import nltk
#     nltk.download('stopwords')
#     nltk.download('punkt')
#
#
# def remove_stopwords(text):
#     from nltk.corpus import stopwords
#     from nltk.tokenize import word_tokenize
#
#     download()
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     filtered_text = ' '.join(filtered_words)
#     return filtered_text
#
#
# def remove_stopwords_chunk(chunks: List[str]) -> List[str]:
#     return [remove_stopwords(a) for a in chunks]
from typing import List

nlp = None  # Initialize nlp as None


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from a given text using spaCy.

    Parameters:
    - text (str): The input text from which stopwords will be removed.

    Returns:
    - str: The text after removing stopwords.
    """
    import spacy

    global nlp
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')  # Load the model only once

    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def remove_stopwords_chunk(chunks: List[str]) -> List[str]:
    """
    Remove stopwords from a list of text chunks using spaCy.

    Parameters:
    - chunks (List[str]): A list of text chunks.

    Returns:
    - List[str]: A list of text chunks with stopwords removed.
    """
    return [remove_stopwords(chunk) for chunk in chunks]
