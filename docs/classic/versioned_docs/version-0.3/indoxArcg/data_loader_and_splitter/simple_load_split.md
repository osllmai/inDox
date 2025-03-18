# Simple Load And Split

The `SimpleLoadAndSplit` function accept both PDF and text files and create chunks based on a semantic text splitter

```python
def __init__(self, file_path: str, remove_sword: bool = False,
             max_chunk_size: Optional[int] = 500, )
```

### Hyperparameters

- file_path (str): The path to the plain text file or PDF file to be processed.
- remove_sword (bool): If True, remove stopwords during the chunking process. Default is False.
- max_chunk_size (Optional[int]): The maximum size of each chunk in characters. Default is 500.

## Usage

To use the SimpleLoadAndSplit function, follow the steps below:

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
from indoxArcg.data_loader_splitter import SimpleLoadAndSplit

file_path = "path/to/your/file.txt"  # Specify the file path
loader_splitter = SimpleLoadAndSplit(file_path=file_path)
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
loader_splitter = ClusteredSplit(file_path=file_path)
docs = loader_splitter.load_and_chunk()
```

This will process the specified file and return all chunks with the
extra clustered layers, forming a hierarchical structure of text chunks.
