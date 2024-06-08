
<p align="center">


<h3 align="center">
    <img src="docs/lite-logo 1.png" alt="Logo" style="width: 80%; opacity: 0.7;"/>

</h3>


<div style="position: relative; width: 100%; text-align: center;">
<h1>
inDox 
<p align="center">Advanced Search and Retrieval Augmentation Generative
</p>

</h1>

[![License](https://img.shields.io/github/license/osllmai/inDox)](https://github.com/osllmai/inDox/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/Indox.svg)](https://pypi.org/project/Indox/0.1.5/)
[![Python](https://img.shields.io/pypi/pyversions/Indox.svg)](https://pypi.org/project/Indox/0.1.4/)
[![Downloads](https://static.pepy.tech/badge/indox)](https://pepy.tech/project/indox)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/inDox?style=social)](https://github.com/osllmai/inDox)




<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://github.com/osllmai/inDox/wiki">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>


<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>



**Indox Retrieval Augmentation** is an innovative application designed to streamline information extraction from a wide
range of document types, including text files, PDF, HTML, Markdown, and LaTeX. Whether structured or unstructured, Indox
provides users with a powerful toolset to efficiently extract relevant data.

Indox Retrieval Augmentation is an innovative application designed to streamline information extraction from a wide
range of document types, including text files, PDF, HTML, Markdown, and LaTeX. Whether structured or unstructured, Indox
provides users with a powerful toolset to efficiently extract relevant data. One of its key features is the ability to
intelligently cluster primary chunks to form more robust groupings, enhancing the quality and relevance of the extracted
information.
With a focus on adaptability and user-centric design, Indox aims to deliver future-ready functionality with more
features planned for upcoming releases. Join us in exploring how Indox can revolutionize your document processing
workflow, bringing clarity and organization to your data retrieval needs.

## Dependency Requirements

Before running this project, ensure that you have the following installed:

- **Python 3.8+**: Required for running the Python backend.
- **PostgreSQL**: Needed if you wish to store your data in a PostgreSQL database.
- **OpenAI API Key**: Necessary if you are using the OpenAI embedding model.
- **HuggingFace API Key**: Necessary if you are using the HuggingFace llms.

Ensure your system also meets these requirements:

- Access to environmental variables for handling sensitive information like API keys.
- Suitable hardware capable of supporting intensive computational tasks.

## Installation


## Getting Started

The following command will install the latest stable inDox

```
pip install Indox
```

To install the latest development version, you may run

```
pip install git+https://github.com/osllmai/inDox@main
```


Clone the repository and navigate to the directory:

```bash
git clone https://github.com/osllmai/inDox.git
cd inDox
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```


### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

# Quick Start

## Setup

### Load Environment Variables

To start, you need to load your API keys from the environment. You can
use either OpenAI or Hugging Face API keys.

``` python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

## Import Indox Package

Import the necessary classes from the Indox package.

``` python
from indox import IndoxRetrievalAugmentation
```

### Importing QA and Embedding Models

``` python
from indox.llms import OpenAiQA
```

``` python
from indox.embeddings import OpenAiEmbedding
```

### Initialize Indox

Create an instance of IndoxRetrievalAugmentation.

``` python
Indox = IndoxRetrievalAugmentation()
```

``` python
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small",openai_api_key=OPENAI_API_KEY)
```


``` python
file_path = "sample.txt"
```

In this section, we take advantage of the `unstructured` library to load
documents and split them into chunks by title. This method helps in
organizing the document into manageable sections for further
processing.

``` python
from indox.data_loader_splitter import UnstructuredLoadAndSplit
```

``` python
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path)
docs = loader_splitter.load_and_chunk()
```

    Starting processing...
    End Chunking process.

Storing document chunks in a vector store is crucial for enabling
efficient retrieval and search operations. By converting text data into
vector representations and storing them in a vector store, you can
perform rapid similarity searches and other vector-based operations.

``` python
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed_openai)
Indox.connect_to_vectorstore(db)
Indox.store_in_vectorstore(docs)
```

    2024-05-14 15:33:04,916 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
    2024-05-14 15:33:12,587 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:33:13,574 - INFO - Document added successfully to the vector store.

    Connection established successfully.

    <Indox.vectorstore.ChromaVectorStore at 0x28cf9369af0>

## Quering

``` python
query = "how cinderella reach her happy ending?"
```

``` python
retriever = indox.QuestionAnswer(vector_database=db,llm=openai_qa,top_k=5)
retriever.invoke(query)
```

    2024-05-14 15:34:55,380 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:35:01,917 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    'Cinderella reached her happy ending by enduring mistreatment from her step-family, finding solace and help from the hazel tree and the little white bird, attending the royal festival where the prince recognized her as the true bride, and ultimately fitting into the golden shoe that proved her identity. This led to her marrying the prince and living happily ever after.'

``` python
retriever.context
```
    ["from the hazel-bush. Cinderella thanked him, went to her mother's\n\ngrave and planted the branch on it, and wept so much that the tears\n\nfell down on it and watered it. And it grew and became a handsome\n\ntree. Thrice a day cinderella went and sat beneath it, and wept and\n\nprayed, and a little white bird always came on the tree, and if\n\ncinderella expressed a wish, the bird threw down to her what she\n\nhad wished for.\n\nIt happened, however, that the king gave orders for a festival",
     'worked till she was weary she had no bed to go to, but had to sleep\n\nby the hearth in the cinders. And as on that account she always\n\nlooked dusty and dirty, they called her cinderella.\n\nIt happened that the father was once going to the fair, and he\n\nasked his two step-daughters what he should bring back for them.\n\nBeautiful dresses, said one, pearls and jewels, said the second.\n\nAnd you, cinderella, said he, what will you have. Father',
     'face he recognized the beautiful maiden who had danced with\n\nhim and cried, that is the true bride. The step-mother and\n\nthe two sisters were horrified and became pale with rage, he,\n\nhowever, took cinderella on his horse and rode away with her. As\n\nthey passed by the hazel-tree, the two white doves cried -\n\nturn and peep, turn and peep,\n\nno blood is in the shoe,\n\nthe shoe is not too small for her,\n\nthe true bride rides with you,\n\nand when they had cried that, the two came flying down and',
     "to send her up to him, but the mother answered, oh, no, she is\n\nmuch too dirty, she cannot show herself. But he absolutely\n\ninsisted on it, and cinderella had to be called. She first\n\nwashed her hands and face clean, and then went and bowed down\n\nbefore the king's son, who gave her the golden shoe. Then she\n\nseated herself on a stool, drew her foot out of the heavy\n\nwooden shoe, and put it into the slipper, which fitted like a\n\nglove. And when she rose up and the king's son looked at her",
     'slippers embroidered with silk and silver. She put on the dress\n\nwith all speed, and went to the wedding. Her step-sisters and the\n\nstep-mother however did not know her, and thought she must be a\n\nforeign princess, for she looked so beautiful in the golden dress.\n\nThey never once thought of cinderella, and believed that she was\n\nsitting at home in the dirt, picking lentils out of the ashes. The\n\nprince approached her, took her by the hand and danced with her.']

```