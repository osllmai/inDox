
# TextPreprocessor

A class for preprocessing text data, including methods for tokenization, stopword removal, stemming, and lemmatization.

## Initialization

The `TextPreprocessor` class is initialized with a list of stopwords and instances of `PorterStemmer` and `WordNetLemmatizer`.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from typing import List
from nltk import pos_tag

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


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
```
## Parameters Explanation

- **stopwords**: A list of stopwords to use for text preprocessing.

## Usage Example

Here is an example of how to use the `TextPreprocessor` class:

```python
from indox.IndoxEval import TextPreprocessor

# Define a sample text"
text = "The quick brown fox jumps over the lazy dog 123."

# Initialize the TextPreprocessor object
preprocessor = TextPreprocessor()

# Preprocess the text
processed_text = preprocessor.preprocess_text(
    text,
    to_lower=True,
    keep_alpha_numeric=True,
    remove_number=True,
    remove_stopword=True,
    stem_word=False,
    lemmatize_word=True,
    top_n_stopwords=5
)

print("Processed Text:", processed_text)
```