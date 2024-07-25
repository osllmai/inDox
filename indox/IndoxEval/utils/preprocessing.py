import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from typing import List
from nltk import pos_tag

# nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# TODO : Handling missing values

with open("indox/IndoxEval/utils/stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()


class TextPreprocessor:
    def __init__(self, stopwords: List[str] = stopwords):
        """
        Initializes the TextPreprocessor with:
        - A set of English stopwords.
        - A Porter Stemmer instance.
        - A WordNet Lemmatizer instance.

        Parameters:
        stopwords (List[str]): A list of stopwords to use for text preprocessing.
        """
        self.stop_words = stopwords
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def to_lower(self, text: str) -> str:
        """
        Converts all characters in the text to lowercase.

        Parameters:
        text (str): The input text to be converted to lowercase.

        Returns:
        str: The text in lowercase.
        """
        return text.lower()

    def keep_alpha_numeric(self, text: str) -> str:
        """
        Removes all non-alphanumeric characters from the text except spaces.
        Replaces multiple spaces with a single space and strips leading/trailing spaces.

        Parameters:
        text (str): The input text to be cleaned.

        Returns:
        str: The cleaned text with only alphanumeric characters and spaces.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_number(self, text: str) -> str:
        """
        Removes all numeric values from the text.

        Parameters:
        text (str): The input text from which numbers will be removed.

        Returns:
        str: The text with numbers removed.
        """
        return re.sub(r"\b\d+\b", "", text)

    def remove_stopword(self, text: List[str], top_n_stopwords: int = 5) -> str:
        """
        Removes stopwords from the text.

        Parameters:
        text (str): The input text from which stopwords will be removed.
        top_n_stopwords (int): The number of top stopwords to be considered for removal.

        Returns:
        str: The text with stopwords removed.
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

        Parameters:
        text (str): The input text to be stemmed.

        Returns:
        str: The text with each word stemmed.
        """
        return " ".join([self.stemmer.stem(word) for word in word_tokenize(text)])

    def get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Maps NLTK part-of-speech tags to WordNet part-of-speech tags.

        Parameters:
        treebank_tag (str): The NLTK part-of-speech tag.

        Returns:
        str: The corresponding WordNet part-of-speech tag.
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_word(self, text: str) -> str:
        """
        Applies lemmatization to each word in the text.

        Parameters:
        text (str): The input text to be lemmatized.

        Returns:
        str: The text with each word lemmatized.
        """
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        return " ".join(
            [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
                for word, pos in tagged_tokens
            ]
        )

    def preprocess_text(
        self,
        text: str,
        to_lower: bool = True,
        keep_alpha_numeric: bool = True,
        remove_number: bool = True,
        remove_stopword: bool = False,
        stem_word: bool = False,
        lemmatize_word: bool = True,
        top_n_stopwords: int = 5,
    ) -> str:
        """
        Applies a list of preprocessing methods to the input text based on the flags provided.

        Parameters:
        text (str): The input text to be processed.
        to_lower (bool): Flag to apply lowercasing.
        keep_alpha_numeric (bool): Flag to remove non-alphanumeric characters.
        remove_number (bool): Flag to remove numbers.
        remove_stopword (bool): Flag to remove stopwords.
        stem_word (bool): Flag to apply stemming.
        lemmatize_word (bool): Flag to apply lemmatization.
        top_n_stopwords (int): The number of top stopwords to be considered for removal.

        Returns:
        str: The processed text after applying all methods.
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
