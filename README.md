<p align="center">


<div style="position: relative; width: 100%; text-align: center;">
    <h1>inDox</h1>
    <a href="https://github.com/osllmai/inDox">
        <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=16&duration=3000&pause=500&multiline=true&width=700&height=100&lines=InDox;Advanced+Search+and+Retrieval+Augmentation+Generative+%7C+Open+Source;Copyright+©️+OSLLAM.ai" alt="Typing SVG" style="margin-top: 20px;"/>
    </a>
</div>



<p align="center">
  <img src="https://raw.githubusercontent.com/osllmai/inDox/master/docs/assets/lite-logo%201.png" alt="inDox Lite Logo">
</p>
</br>

[![License](https://img.shields.io/github/license/osllmai/inDox)](https://github.com/osllmai/inDox/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/Indox.svg)](https://pypi.org/project/Indox/0.1.26/)
[![Python](https://img.shields.io/pypi/pyversions/Indox.svg)](https://pypi.org/project/Indox/0.1.8/)
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

## Roadmap
| 🤖 Model Support          | Implemented | Description                                           |
|---------------------------|-------------|-------------------------------------------------------|
| Ollama (e.g. Llama3)      | ✅           | Local Embedding and LLM Models powered by Ollama      |
| HuggingFace               | ✅           | Local Embedding and LLM Models powered by HuggingFace |
| Mistral  | ✅           | Embedding and LLM Models by Cohere                    |
| Google (e.g. Gemini)      | ✅           | Embedding and Generation Models by Google             |
| OpenAI (e.g. GPT4)        | ✅           | Embedding and Generation Models by OpenAI 

| Supported Model Via Indox Api | Implemented | Description                                    |
|-------------------------------|-------------|------------------------------------------------|
| OpenAi                        | ✅           | Embedding and LLm OpenAi Model From Indox Api  |
| Mistral                       | ✅           | Embedding and LLm Mistral Model From Indox Api |
| Anthropic                     | ❌           |     Embedding and LLm Anthropic Model From Indox Api |                                          |

| 📁 Loader and Splitter  | Implemented | Description                                      |
|--------------------------------| ----------- |--------------------------------------------------|
| Simple PDF                     | ✅          | Import PDF                                       |
| UnstructuredIO                 | ✅          | Import Data through Unstructured                 |
|Clustered Load And Split|✅| Load pdf and texts. add a extra clustering layer |

| ✨ RAG Features        | Implemented | Description                                                           |
|-----------------------|-------------|-----------------------------------------------------------------------|
| Hybrid Search         | ❌           | Semantic Search combined with Keyword Search                          |
| Semantic Caching      | ✅           | Results saved and retrieved based on semantic meaning                 |
| Clustered Prompt      | ✅           | Retrieve smaller chunks and do clustering and summarization           |
| Agentic Rag           | ✅  | Generate more reliabale answer, rank context and web search if needed |
| Advanced Querying     | ❌  | Task Delegation Based on LLM Evaluation                               |
| Reranking             | ✅           | Rerank results based on context for improved results                  |
| Customizable Metadata | ❌  | Free control over Metadata                                            |

| 🆒 Cool Bonus         | Implemented | Description                                             |
| --------------------- | ----------- |---------------------------------------------------------|
| Docker Support        | ❌         | Indox is deployable via Docker                          |
| Customizable Frontend | ❌          | Indox's frontend is fully-customizable via the frontend |


## Examples
| ☑️ Examples                    | Run in Colab                                                                                                                                                                        | 
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Indox Api (OpenAi)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/indox_api_openai.ipynb)        |
| Mistral (Using Unstructured)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/mistral_unstructured.ipynb)    |
| OpenAi (Using Clustered Split) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/openai_clusterSplit.ipynb)     |
| HuggingFace Models(Mistral)    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/hf_mistral_SimpleReader.ipynb) |
| Ollama                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/ollama.ipynb)                  |






## Indox Workflow
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/osllmai/inDox/master/docs/assets/inDox.png" alt="inDox work flow" width="80%">
</div>


## Getting Started

The following command will install the latest stable inDox

```
pip install Indox
```

To install the latest development version, you may run

```
pip install git+https://github.com/osllmai/inDox@master
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

## Setting Up the Python Environment

If you are running this project in your local IDE, please create a Python environment to ensure all dependencies are correctly managed. You can follow the steps below to set up a virtual environment named `indox`:

### Windows

1. **Create the virtual environment:**
```bash
  python -m venv indox
```

2. **Activate the virtual environment:**
```bash
  indox\Scripts\activate
```


### macOS/Linux

1. **Create the virtual environment:**
   ```bash
   python3 -m venv indox
   
2. **Activate the virtual environment:**
```bash
  source indox/bin/activate
```

### Install Dependencies

Once the virtual environment is activated, install the required dependencies by running:

```bash
  pip install -r requirements.txt
```


### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load LLM And Embedding Models**: Initialize your embedding model from Indox's selection of pre-trained models.

# Quick Start

### Install the Required Packages

```bash
pip install indox
pip install openai
pip install chromadb
```

## Setting Up the Python Environment

If you are running this project in your local IDE, please create a Python environment to ensure all dependencies are correctly managed. You can follow the steps below to set up a virtual environment named `indox`:

### Windows

1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
indox_judge\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**
   ```bash
   python3 -m venv indox
```

2. **Activate the virtual environment:**
    ```bash
   source indox/bin/activate
```
### Install Dependencies

Once the virtual environment is activated, install the required dependencies by running:

```bash
pip install -r requirements.txt
```


### Load Environment Variables

To start, you need to load your API keys from the environment.

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

### Importing LLM and Embedding Models

``` python
from indox.llms import OpenAi
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





```txt
  .----------------.  .-----------------. .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |     _____    | || | ____  _____  | || |  ________    | || |     ____     | || |  ____  ____  | |
| |    |_   _|   | || ||_   \|_   _| | || | |_   ___ `.  | || |   .'    `.   | || | |_  _||_  _| | |
| |      | |     | || |  |   \ | |   | || |   | |   `. \ | || |  /  .--.  \  | || |   \ \  / /   | |
| |      | |     | || |  | |\ \| |   | || |   | |    | | | || |  | |    | |  | || |    > `' <    | |
| |     _| |_    | || | _| |_\   |_  | || |  _| |___.' / | || |  \  `--'  /  | || |  _/ /'`\ \_  | |
| |    |_____|   | || ||_____|\____| | || | |________.'  | || |   `.____.'   | || | |____||____| | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
```



