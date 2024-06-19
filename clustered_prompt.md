# Clustered\_Prompt

The `generate_clustered_prompts` function Clusters the provided context using the given embeddings and generates clustered prompts and returns a list of clustered document segments based on the provided context and embeddings. This function Sets the `cluster_prompt` attribute to True.

```python
def generate_clustered_prompts(context, embedings):
```

#### Parameters:

* context (list of str): A list of text strings to be clustered.
* embeddings: Embeddings function.

### Usage

To use the `generate_clustered_prompts` function, follow the steps below:

Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

Perform the clustered split on the text file or PDF file:

```python
from indox.data_loader_splitter import ClusteredSplit

file_path = "path/to/your/file.txt"  # Specify the file path
use_openai_summary = os.getenv("OPENAI_API_KEY") # Specify openai API

chunks = ClusteredSplit(file_path=file_path, embeddings=embeddings, chunk_size=50, threshold=0.1, dim=30,use_openai_summary=use_openai_summary)
loader_splitter.cluster_prompt = True
docs = loader_splitter.load_and_chunk()
```
