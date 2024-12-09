[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDoxRag/blob/master/Demo/quick_start.ipynb)

# Quick Start

## Overview

This documentation provides a detailed explanation of how to use the
`IndoxRetrievalAugmentation` package for QA model and embedding
selection, document splitting, and storing in a vector store.

## Setup

### Install the Required Packages

```python
!pip install indoxRag
!pip install openai
!pip install chromadb
```

### Load Environment Variables

To start, you need to load your API keys from the environment.

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

## Import IndoxRag Package

Import the necessary classes from the IndoxRag package.

```python
from indoxRag import IndoxRetrievalAugmentation
```

### Importing LLM and Embedding Models

```python
from indoxRag.llms import OpenAi
```

```python
from indoxRag.embeddings import OpenAiEmbedding
```

### Initialize IndoxRag

Create an instance of IndoxRetrievalAugmentation.

```python
IndoxRag = IndoxRetrievalAugmentation()
```

```python
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small",openai_api_key=OPENAI_API_KEY)
```

```python
file_path = "sample.txt"
```

In this section, we take advantage of the `unstructured` library to load
documents and split them into chunks by title. This method helps in
organizing the document into manageable sections for further
processing.

```python
from indoxRag.data_loader_splitter import UnstructuredLoadAndSplit
```

```python
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path)
docs = loader_splitter.load_and_chunk()
```

    Starting processing...
    End Chunking process.

Storing document chunks in a vector store is crucial for enabling
efficient retrieval and search operations. By converting text data into
vector representations and storing them in a vector store, you can
perform rapid similarity searches and other vector-based operations.

```python
from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed_openai)
IndoxRag.connect_to_vectorstore(db)
IndoxRag.store_in_vectorstore(docs)
```

    2024-05-14 15:33:04,916 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
    2024-05-14 15:33:12,587 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:33:13,574 - INFO - Document added successfully to the vector store.

    Connection established successfully.

    <IndoxRag.vectorstore.ChromaVectorStore at 0x28cf9369af0>

## Quering

```python
query = "how cinderella reach her happy ending?"
```

```python
retriever = indoxRag.QuestionAnswer(vector_database=db,llm=openai_qa,top_k=5)
retriever.invoke(query)
```

    2024-05-14 15:34:55,380 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:35:01,917 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    'Cinderella reached her happy ending by enduring mistreatment from her step-family, finding solace and help from the hazel tree and the little white bird, attending the royal festival where the prince recognized her as the true bride, and ultimately fitting into the golden shoe that proved her identity. This led to her marrying the prince and living happily ever after.'

```python
retriever.context

```

    ["from the hazel-bush. Cinderella thanked him, went to her mother's\n\ngrave and planted the branch on it, and wept so much that the tears\n\nfell down on it and watered it. And it grew and became a handsome\n\ntree. Thrice a day cinderella went and sat beneath it, and wept and\n\nprayed, and a little white bird always came on the tree, and if\n\ncinderella expressed a wish, the bird threw down to her what she\n\nhad wished for.\n\nIt happened, however, that the king gave orders for a festival",
     'worked till she was weary she had no bed to go to, but had to sleep\n\nby the hearth in the cinders. And as on that account she always\n\nlooked dusty and dirty, they called her cinderella.\n\nIt happened that the father was once going to the fair, and he\n\nasked his two step-daughters what he should bring back for them.\n\nBeautiful dresses, said one, pearls and jewels, said the second.\n\nAnd you, cinderella, said he, what will you have. Father',
     'face he recognized the beautiful maiden who had danced with\n\nhim and cried, that is the true bride. The step-mother and\n\nthe two sisters were horrified and became pale with rage, he,\n\nhowever, took cinderella on his horse and rode away with her. As\n\nthey passed by the hazel-tree, the two white doves cried -\n\nturn and peep, turn and peep,\n\nno blood is in the shoe,\n\nthe shoe is not too small for her,\n\nthe true bride rides with you,\n\nand when they had cried that, the two came flying down and',
     "to send her up to him, but the mother answered, oh, no, she is\n\nmuch too dirty, she cannot show herself. But he absolutely\n\ninsisted on it, and cinderella had to be called. She first\n\nwashed her hands and face clean, and then went and bowed down\n\nbefore the king's son, who gave her the golden shoe. Then she\n\nseated herself on a stool, drew her foot out of the heavy\n\nwooden shoe, and put it into the slipper, which fitted like a\n\nglove. And when she rose up and the king's son looked at her",
     'slippers embroidered with silk and silver. She put on the dress\n\nwith all speed, and went to the wedding. Her step-sisters and the\n\nstep-mother however did not know her, and thought she must be a\n\nforeign princess, for she looked so beautiful in the golden dress.\n\nThey never once thought of cinderella, and believed that she was\n\nsitting at home in the dirt, picking lentils out of the ashes. The\n\nprince approached her, took her by the hand and danced with her.']
