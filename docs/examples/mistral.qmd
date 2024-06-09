---
title: Mistral As Question Answer Model
---
## Introduction

In this notebook, we will demonstrate how to securely handle `inDox` as system for question answering system with open source models which are available on internet like `Mistral`. so firstly you should buil environment variables and API keys in Python using the `dotenv` library. Environment variables are a crucial part of configuring your applications, especially when dealing with sensitive information like API keys.

::: {.callout-note}
Because we are using **HuggingFace** models you need to define your `HUGGINGFACE_API_KEY` in `.env` file. This allows us to keep our API keys and other sensitive information out of our codebase, enhancing security and maintainability.
:::

Let's start by importing the required libraries and loading our environment variables.



```python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
```

### Import Essential Libraries 
Then, we import essential libraries for our `Indox` question answering system:
- `IndoxRetrievalAugmentation`: Enhances the retrieval process for better QA performance.
- `MistralQA`: A powerful QA model from Indox, built on top of the Hugging Face model.
- `HuggingFaceEmbedding`: Utilizes Hugging Face embeddings for improved semantic understanding.
- `UnstructuredLoadAndSplit`: A utility for loading and splitting unstructured data.


```python
from indox import IndoxRetrievalAugmentation
from indox.llms import MistralQA
from indox.embeddings import HuggingFaceEmbedding
from indox.data_loader_splitter import UnstructuredLoadAndSplit
```

### Building the Indox System and Initializing Models

Next, we will build our `inDox` system and initialize the Mistral question answering model along with the embedding model. This setup will allow us to leverage the advanced capabilities of Indox for our question answering tasks.



```python
indox = IndoxRetrievalAugmentation()
mistral_qa = MistralQA(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1")
```

### Setting Up Reference Directory and File Path

To demonstrate the capabilities of our Indox question answering system, we will use a sample directory. This directory will contain our reference data, which we will use for testing and evaluation.

First, we specify the path to our sample file. In this case, we are using a file named `sample.txt` located in our working directory. This file will serve as our reference data for the subsequent steps.

Let's define the file path for our reference data.


```python
file_path = "sample.txt"
```

### Chunking Reference Data with UnstructuredLoadAndSplit

To effectively utilize our reference data, we need to process and chunk it into manageable parts. This ensures that our question answering system can efficiently handle and retrieve relevant information.

We use the `UnstructuredLoadAndSplit` utility for this task. This tool allows us to load the unstructured data from our specified file and split it into smaller chunks. This process enhances the performance of our retrieval and QA models by making the data more accessible and easier to process.

In this step, we define the file path for our reference data and use `UnstructuredLoadAndSplit` to chunk the data with a maximum chunk size of 400 characters.

Let's proceed with chunking our reference data.



```python
load_splitter = UnstructuredLoadAndSplit(file_path=file_path,max_chunk_size=400)
docs = load_splitter.load_and_chunk()
```

### Connecting Embedding Model to Indox

With our reference data chunked and ready, the next step is to connect our embedding model to the Indox system. This connection enables the system to leverage the embeddings for better semantic understanding and retrieval performance.

We use the `connect_to_vectorstore` method to link the `HuggingFaceEmbedding` model with our Indox system. By specifying the embeddings and a collection name, we ensure that our reference data is appropriately indexed and stored, facilitating efficient retrieval during the question-answering process.

Let's connect the embedding model to Indox.



```python
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed)
```


```python
indox.connect_to_vectorstore(vectorstore_database=db)
```




    <indox.vector_stores.Chroma.ChromaVectorStore at 0x146b850ddc0>



### Storing Data in the Vector Store

After connecting our embedding model to the Indox system, the next step is to store our chunked reference data in the vector store. This process ensures that our data is indexed and readily available for retrieval during the question-answering process.

We use the `store_in_vectorstore` method to store the processed data in the vector store. By doing this, we enhance the system's ability to quickly access and retrieve relevant information based on the embeddings generated earlier.

Let's proceed with storing the data in the vector store.



```python
indox.store_in_vectorstore(docs)
```




    <indox.vector_stores.Chroma.ChromaVectorStore at 0x146b850ddc0>



## Query from RAG System with Indox
With our Retrieval-Augmented Generation (RAG) system built using Indox, we are now ready to test it with a sample question. This test will demonstrate how effectively our system can retrieve and generate accurate answers based on the reference data stored in the vector store.

We'll use a sample query to test our system:
- **Query**: "How did Cinderella reach her happy ending?"

This question will be processed by our Indox system to retrieve relevant information and generate an appropriate response.

Let's test our RAG system with the sample question


```python
query = "How cinderella reach her happy ending?"
```

Now that our Retrieval-Augmented Generation (RAG) system with Indox is fully set up, we can test it with a sample question. We'll use the `invoke` submethod to get a response from the system.


The `invoke` method processes the query using the connected QA model and retrieves relevant information from the vector store. It returns a list where:
- The first index contains the answer.
- The second index contains the contexts and their respective scores.


We'll pass this query to the `invoke` method and print the response.



```python
retriever = indox.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5)
```


```python
answer = retriever.invoke(query=query)
```


```python
context = retriever.context
context
```




    ['by the hearth in the cinders. And as on that account she always\n\nlooked dusty and dirty, they called her cinderella.\n\nIt happened that the father was once going to the fair, and he\n\nasked his two step-daughters what he should bring back for them.\n\nBeautiful dresses, said one, pearls and jewels, said the second.\n\nAnd you, cinderella, said he, what will you have. Father',
     'cinderella expressed a wish, the bird threw down to her what she\n\nhad wished for.\n\nIt happened, however, that the king gave orders for a festival\n\nwhich was to last three days, and to which all the beautiful young\n\ngirls in the country were invited, in order that his son might choose\n\nhimself a bride. When the two step-sisters heard that they too were',
     'know where she was gone. He waited until her father came, and\n\nsaid to him, the unknown maiden has escaped from me, and I\n\nbelieve she has climbed up the pear-tree. The father thought,\n\ncan it be cinderella. And had an axe brought and cut the\n\ntree down, but no one was on it. And when they got into the\n\nkitchen, cinderella lay there among the ashes, as usual, for she',
     'and had run to the little hazel-tree, and there she had taken off\n\nher beautiful clothes and laid them on the grave, and the bird had\n\ntaken them away again, and then she had seated herself in the\n\nkitchen amongst the ashes in her grey gown.\n\nNext day when the festival began afresh, and her parents and\n\nthe step-sisters had gone once more, cinderella went to the\n\nhazel-tree and said -',
     "had jumped down on the other side of the tree, had taken the\n\nbeautiful dress to the bird on the little hazel-tree, and put on her\n\ngrey gown.\n\nOn the third day, when the parents and sisters had gone away,\n\ncinderella went once more to her mother's grave and said to the"]



## Evaluation
Evaluating the performance of your question-answering system is crucial to ensure the quality and reliability of the responses. In this section, we will use the `Evaluation` module from Indox to assess our system's outputs.



```python
from indox.evaluation import Evaluation
evaluator = Evaluation(["BertScore", "Toxicity"])
```

### Preparing Inputs for Evaluation
Next, we need to format the inputs according to the Indox evaluator's requirements. This involves creating a dictionary that includes the question, the generated answer, and the context from which the answer was derived.


```python
inputs = {
    "question" : query,
    "answer" : answer,
    "context" : context
}
result = evaluator(inputs)
```


```python
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision</th>
      <td>0.524382</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.537209</td>
    </tr>
    <tr>
      <th>F1-score</th>
      <td>0.530718</td>
    </tr>
    <tr>
      <th>Toxicity</th>
      <td>0.074495</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
