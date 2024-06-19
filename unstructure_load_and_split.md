# Unstructure\_Load\_And\_Split

The UnstructuredLoadAndSplit function uses the unstructured library to import various file types and split them into chunks. By default, it uses the “split by title” method from the unstructured library, but users can also choose the semantic\_text\_splitter.

```python
def UnstructuredLoadAndSplit(file_path: str,
                             remove_sword: bool = False,
                             max_chunk_size: int = 500,
                             splitter=None)
```

#### Hyperparameters

* file\_path (str): The path to the file to be processed. Various file types are supported.
* remove\_sword (bool): If True, remove stop words during the chunking process. Default is False.
* max\_chunk\_size (int): The maximum size of each chunk in characters. Default is 500.
* splitter: The method used to split the text. The default is “split by title” from the unstructured library. Users can also choose semantic\_text\_splitter.

### Usage

To use the UnstructuredLoadAndSplit function, follow the steps below:

Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

Initialize Indox and QA models:

```python
from indox import IndoxRetrievalAugmentation
Indox = IndoxRetrievalAugmentation()

from indox.qa_models import OpenAiQA
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
```

Perform the unstructured load and split on the file:

```python
from indox.data_loader_splitter import UnstructuredLoadAndSplit
from indox.splitter import semantic_text_splitter

file_path = "path/to/your/file.pdf"  # Specify the file path
docs = UnstructuredLoadAndSplit(file_path=file_path,
                                remove_sword=False,
                                max_chunk_size=500,
                                splitter=semantic_text_splitter)
```

### Example Code

Here’s a complete example of using the UnstructuredLoadAndSplit function in a Jupyter notebook:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

from indox import IndoxRetrievalAugmentation
Indox = IndoxRetrievalAugmentation()

from indox.qa_models import OpenAiQA
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

from indox.data_loader_splitter import UnstructuredLoadAndSplit
from indox.splitter import semantic_text_splitter

file_path = "path/to/your/file.pdf"  # Specify the file path
docs = UnstructuredLoadAndSplit(file_path=file_path,
                                remove_sword=False,
                                max_chunk_size=500,
                                splitter=semantic_text_splitter)
```

***

Previous: Cluster Split | Next: Embedding Models
