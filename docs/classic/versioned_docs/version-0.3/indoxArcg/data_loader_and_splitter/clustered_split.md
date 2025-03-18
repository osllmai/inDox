# Clustered Split

The `ClusteredSplit` function creates leaf chunks from the text and adds
extra clustered chunks to these leaf chunks. The clustering continues
until no new clusters are available, growing like a tree: starting from
leaf chunks, then clustering between the last clustered chunks, and so
on.

```python
def __init__(self, file_path: str, embeddings, re_chunk: bool = False, remove_sword: bool = False,
             chunk_size: Optional[int] = 100, overlap: Optional[int] = 0, threshold: float = 0.1, dim: int = 10,
             use_openai_summary: bool = False, max_len_summary: int = 100, min_len_summary: int = 30)
```

### Hyperparameters

- file_path (str): The path to the plain text file or PDF file to be processed.
- embeddings: The embeddings to be used for clustering.
- re_chunk (bool): If True, re-chunk the text after initial chunking. Default is False.
- remove_sword (bool): If True, remove stopwords during the chunking process. Default is False.
- chunk_size (Optional[int]): The size of each chunk in characters. Default is 100.
- overlap (Optional[int]): The number of characters to overlap between chunks. Default is 0.
- threshold (float): The similarity threshold for creating clusters. Default is 0.1.
- dim (int): The dimensionality of the embeddings. Default is 10.
- use_openai_summary (bool, optional): Whether to use OpenAI summary for summarizing the chunks. Default is False.
- max_len_summary (int, optional): The maximum length of the summary. Default is 100.
- min_len_summary (int, optional): The minimum length of the summary. Default is 30.

## Usage

To use the ClusteredSplit function, follow the steps below:

Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

Initialize indoxArcg and QA models:

```python
from indoxArcg.llms import OpenAiQA
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="GPT4-o")
```

Perform the clustered split on the text file or PDF file:

```python
from indoxArcg.data_loader_splitter import ClusteredSplit

file_path = "path/to/your/file.txt"  # Specify the file path
loader_splitter = ClusteredSplit(file_path=file_path)
docs = loader_splitter.load_and_chunk()
```

## Example Code

Here’s a complete example of using the ClusteredSplit function in a
Jupyter notebook:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

from indoxArcg import indoxArcgRetrievalAugmentation
indoxArcg = indoxArcgRetrievalAugmentation()

from indoxArcg.llms import OpenAiQA
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="GPT4-o")

from indoxArcg.data_loader_splitter import ClusteredSplit

file_path = "path/to/your/file.txt"  # Specify the file path
loader_splitter = ClusteredSplit(file_path=file_path,
                        embeddings=openai_qa.embeddings,
                        re_chunk=False,
                        remove_sword=False,
                        chunk_size=100,
                        overlap=0,
                        threshold=0.1,
                        dim=10)
docs = loader_splitter.load_and_chunk()
```

This will process the specified file and return all chunks with the
extra clustered layers, forming a hierarchical structure of text chunks.
