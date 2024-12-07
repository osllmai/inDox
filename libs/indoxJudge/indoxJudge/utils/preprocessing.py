import re
from typing import List

stopwords = [
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they", "I",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
    "but", "not", "what", "all", "were", "we", "when", "your", "can",
    "said", "there", "use", "an", "each", "which", "she", "do", "how",
    "their", "if"
]


class TextPreprocessor:
    stemmer = None
    lemmatizer = None
    _initialized = False

    def __init__(self, stopwords: List[str] = stopwords):
        """
        Initializes the TextPreprocessor with:
        - A set of English stopwords.
        - A Porter Stemmer instance.
        - A WordNet Lemmatizer instance.

        Parameters:
        stopwords (List[str]): A list of stopwords to use for text preprocessing.
        """
        self._initialize_class_resources()
        self.stop_words = stopwords

    @classmethod
    def _initialize_class_resources(cls):
        """Initialize class-level resources (stemmer, lemmatizer) only once."""
        if not cls._initialized:
            from indoxJudge.utils import nltk_download
            from nltk.stem import PorterStemmer
            from nltk.stem import WordNetLemmatizer

            nltk_download()
            cls.stemmer = PorterStemmer()
            cls.lemmatizer = WordNetLemmatizer()
            cls._initialized = True

    def to_lower(self, text: str) -> str:
        return text.lower()

    def keep_alpha_numeric(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_number(self, text: str) -> str:
        return re.sub(r"\b\d+\b", "", text)

    def remove_stopword(self, text: str, top_n_stopwords: int = 5) -> str:
        from nltk.tokenize import word_tokenize

        self.stop_words = self.stop_words[0:top_n_stopwords]
        return " ".join(
            [
                word
                for word in word_tokenize(text)
                if word.lower() not in self.stop_words
            ]
        )

    def stem_word(self, text: str) -> str:
        from nltk.tokenize import word_tokenize

        return " ".join([self.stemmer.stem(word) for word in word_tokenize(text)])

    def get_wordnet_pos(self, treebank_tag: str) -> str:
        from nltk.corpus import wordnet

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
        from nltk import pos_tag
        from nltk.tokenize import word_tokenize

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
