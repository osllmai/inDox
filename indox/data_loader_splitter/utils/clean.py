from typing import List


def download():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')


def remove_stopwords(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    download()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def remove_stopwords_chunk(chunks: List[str]) -> List[str]:
    return [remove_stopwords(a) for a in chunks]
