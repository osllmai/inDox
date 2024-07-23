import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List

# nltk.download("stopwords")

# TODO : Handling missing values

with open("utils/stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()


class TextPreprocessor:
    def __init__(self, stopwords: List[str] = stopwords):
        """
        Initializes the TextPreprocessor with:
        - A set of English stopwords
        - A Porter Stemmer instance
        - A WordNet Lemmatizer instance
        """
        self.stop_words = stopwords
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

    def remove_stopword(self, text: List[str], top_n_stopwords: int = 5) -> str:
        """
        Removes stopwords from the text.

        :param text: The input text from which stopwords will be removed.
        :return: The text with stopwords removed.
        """
        self.stop_words = self.stop_words[0:top_n_stopwords]
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

    def preprocess_text(
        self,
        text: str,
        to_lower: bool = True,
        keep_alpha_numeric: bool = True,
        remove_number: bool = True,
        remove_stopword: bool = True,
        stem_word: bool = False,
        lemmatize_word: bool = True,
        top_n_stopwords: int = 5,
    ) -> str:
        """
        Applies a list of preprocessing methods to the input text based on the flags provided.

        :param text: The input text to be processed.
        :param to_lower: Flag to apply lowercasing.
        :param keep_alpha_numeric: Flag to remove non-alphanumeric characters.
        :param remove_number: Flag to remove numbers.
        :param remove_stopword: Flag to remove stopwords.
        :param stem_word: Flag to apply stemming.
        :param lemmatize_word: Flag to apply lemmatization.
        :return: The processed text after applying all methods.
        """
        if to_lower:
            text = self.to_lower(text)
        if keep_alpha_numeric:
            text = self.keep_alpha_numeric(text)
        if remove_number:
            text = self.remove_number(text)
        if remove_stopword:
            text = self.remove_stopword(text, top_n_stopwords)
        if stem_word:
            text = self.stem_word(text)
        if lemmatize_word:
            text = self.lemmatize_word(text)
        return text
