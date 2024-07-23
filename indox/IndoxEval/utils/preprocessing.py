import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Callable

nltk.download("stopwords")

# TODO : Handling missing values
# TODO : Remove top n stopwords


class TextPreprocessor:
    def __init__(self):
        """
        Initializes the TextPreprocessor with:
        - A set of English stopwords
        - A Porter Stemmer instance
        - A WordNet Lemmatizer instance
        - A SpellChecker instance for spelling correction
        """
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def to_lower(self, text: str) -> str:
        """
        Converts all characters in the text to lowercase.

        :param text: The input text to be converted to lowercase.
        :return: The text in lowercase.
        """
        return text.lower()

    def keep_alpha_numeric(self, text: str) -> str:
        """
        Removes all non-alphanumeric characters from the text except spaces.
        Replaces multiple spaces with a single space and strips leading/trailing spaces.

        :param text: The input text to be cleaned.
        :return: The cleaned text with only alphanumeric characters and spaces.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_number(self, text: str) -> str:
        """
        Removes all numeric values from the text.

        :param text: The input text from which numbers will be removed.
        :return: The text with numbers removed.
        """
        return re.sub(r"\b\d+\b", "", text)

    def remove_stopword(self, text: str) -> str:
        """
        Removes stopwords from the text.

        :param text: The input text from which stopwords will be removed.
        :return: The text with stopwords removed.
        """
        return " ".join(
            [
                word
                for word in word_tokenize(text)
                if word.lower() not in self.stop_words
            ]
        )

    def stem_word(self, text: str) -> str:
        """
        Applies stemming to each word in the text.

        :param text: The input text to be stemmed.
        :return: The text with each word stemmed.
        """
        return " ".join([self.stemmer.stem(word) for word in word_tokenize(text)])

    def lemmatize_word(self, text: str) -> str:
        """
        Applies lemmatization to each word in the text.

        :param text: The input text to be lemmatized.
        :return: The text with each word lemmatized.
        """
        return " ".join(
            [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        )

    def preprocess_text(self, text: str, methods: List[Callable[[str], str]]) -> str:
        """
        Applies a list of preprocessing methods to the input text.

        :param text: The input text to be processed.
        :param methods: A list of preprocessing methods (functions) to be applied to the text.
        :return: The processed text after applying all methods.
        """
        for method in methods:
            text = method(text)
        return text
