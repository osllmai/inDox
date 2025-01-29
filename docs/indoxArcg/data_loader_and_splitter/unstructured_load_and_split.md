## UnstructuredLoadAndSplit

The UnstructuredLoadAndSplit function uses the unstructured library to
import various file types and split them into chunks. By default, it
uses the “split by title” method from the unstructured library, but
users can also choose the semantic_text_splitter.

```python
def UnstructuredLoadAndSplit(file_path: str,
                             remove_sword: bool = False,
                             max_chunk_size: int = 500,
                             splitter=None)
```

### Hyperparameters

- file_path (str): The path to the file to be processed. Various file
  types are supported.
- remove_sword (bool): If True, remove stop words during the chunking
  process. Default is False.
- max_chunk_size (int): The maximum size of each chunk in characters.
  Default is 500.
- splitter: The method used to split the text. The default is “split
  by title” from the unstructured library. Users can also choose
  semantic_text_splitter.

## Usage

To use the UnstructuredLoadAndSplit function, follow the steps below:

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
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
```

Perform the unstructured load and split on the file:

```python
from indoxArcg.data_loader_splitter import UnstructuredLoadAndSplit
from indoxArcg.splitter import semantic_text_splitter

file_path = "path/to/your/file.pdf"  # Specify the file path
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path,
                                remove_sword=False,
                                max_chunk_size=500,
                                splitter=semantic_text_splitter)
docs = loader_splitter.load_and_chunk()
```

## Example Code

Here’s a complete example of using the UnstructuredLoadAndSplit function
in a Jupyter notebook:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

from indoxArcg import indoxArcgRetrievalAugmentation
indoxArcg = indoxArcgRetrievalAugmentation()

from indoxArcg.llms import OpenAiQA
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

from indoxArcg.data_loader_splitter import UnstructuredLoadAndSplit
from indoxArcg.splitter import semantic_text_splitter

file_path = "path/to/your/file.pdf"  # Specify the file path
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path,
                                remove_sword=False,
                                max_chunk_size=500,
                                splitter=semantic_text_splitter)
docs = loader_splitter.load_and_chunk()
```
