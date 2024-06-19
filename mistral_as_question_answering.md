# Mistral\_as\_Question\_Answering

### Environment Variables and API Keys

In this notebook, we will demonstrate how to securely handle `inDox` as system for question answering system with open source models which are available on internet like `Mistral`. so firstly you should buil environment variables and API keys in Python using the `dotenv` library. Environment variables are a crucial part of configuring your applications, especially when dealing with sensitive information like API keys.

::: {.callout-note} Because we are using **HuggingFace** models you need to define your `HF_API_KEY` in `.env` file. This allows us to keep our API keys and other sensitive information out of our codebase, enhancing security and maintainability. ::: Let's start by importing the required libraries and loading our environment variables.

```{python}
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv('HF_API_KEY')
```

#### Import Essential Libraries

Then, we import essential libraries for our `Indox` question answering system:

* `IndoxRetrievalAugmentation`: Enhances the retrieval process for better QA performance.
* `MistralQA`: A powerful QA model from Indox, built on top of the Hugging Face model.
* `HuggingFaceEmbedding`: Utilizes Hugging Face embeddings for improved semantic understanding.
* `UnstructuredLoadAndSplit`: A utility for loading and splitting unstructured data.

```{python}
#| ExecuteTime: {end_time: '2024-05-20T13:46:46.343004Z', start_time: '2024-05-20T13:46:09.122327Z'}
from indox import IndoxRetrievalAugmentation
from indox.qa_models import MistralQA
from indox.embeddings import HuggingFaceEmbedding
from indox.data_loader_splitter import UnstructuredLoadAndSplit
```

#### Building the Indox System and Initializing Models

Next, we will build our `inDox` system and initialize the Mistral question answering model along with the embedding model. This setup will allow us to leverage the advanced capabilities of Indox for our question answering tasks.

```{python}
#| ExecuteTime: {end_time: '2024-05-20T13:46:48.029295Z', start_time: '2024-05-20T13:46:46.343004Z'}
indox = IndoxRetrievalAugmentation()
mistral_qa = MistralQA(api_key=HF_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding()
```

#### Setting Up Reference Directory and File Path

To demonstrate the capabilities of our Indox question answering system, we will use a sample directory. This directory will contain our reference data, which we will use for testing and evaluation.

First, we specify the path to our sample file. In this case, we are using a file named `sample.txt` located in our working directory. This file will serve as our reference data for the subsequent steps.

Let's define the file path for our reference data.

```{python}
file_path = "sample.txt"
```

#### Chunking Reference Data with UnstructuredLoadAndSplit

To effectively utilize our reference data, we need to process and chunk it into manageable parts. This ensures that our question answering system can efficiently handle and retrieve relevant information.

We use the `UnstructuredLoadAndSplit` utility for this task. This tool allows us to load the unstructured data from our specified file and split it into smaller chunks. This process enhances the performance of our retrieval and QA models by making the data more accessible and easier to process.

In this step, we define the file path for our reference data and use `UnstructuredLoadAndSplit` to chunk the data with a maximum chunk size of 400 characters.

Let's proceed with chunking our reference data.

```{python}
data = UnstructuredLoadAndSplit(file_path=file_path,max_chunk_size=400)
```

#### Connecting Embedding Model to Indox

With our reference data chunked and ready, the next step is to connect our embedding model to the Indox system. This connection enables the system to leverage the embeddings for better semantic understanding and retrieval performance.

We use the `connect_to_vectorstore` method to link the `HuggingFaceEmbedding` model with our Indox system. By specifying the embeddings and a collection name, we ensure that our reference data is appropriately indexed and stored, facilitating efficient retrieval during the question-answering process.

Let's connect the embedding model to Indox.

```{python}
indox.connect_to_vectorstore(embeddings=embed,collection_name="sample")
```

#### Storing Data in the Vector Store

After connecting our embedding model to the Indox system, the next step is to store our chunked reference data in the vector store. This process ensures that our data is indexed and readily available for retrieval during the question-answering process.

We use the `store_in_vectorstore` method to store the processed data in the vector store. By doing this, we enhance the system's ability to quickly access and retrieve relevant information based on the embeddings generated earlier.

Let's proceed with storing the data in the vector store.

```{python}
indox.store_in_vectorstore(data)
```

### Testing the RAG System with Indox

With our Retrieval-Augmented Generation (RAG) system built using Indox, we are now ready to test it with a sample question. This test will demonstrate how effectively our system can retrieve and generate accurate answers based on the reference data stored in the vector store.

We'll use a sample query to test our system:

* **Query**: "How did Cinderella reach her happy ending?"

This question will be processed by our Indox system to retrieve relevant information and generate an appropriate response.

Let's test our RAG system with the sample question.

```{python}
query = "How cinderella reach her happy ending?"
```

Now that our Retrieval-Augmented Generation (RAG) system with Indox is fully set up, we can test it with a sample question. We'll use the `answer_question` submethod to get a response from the system.

::: {.callout-note}

The `answer_question` method processes the query using the connected QA model and retrieves relevant information from the vector store. It returns a list where:

* The first index contains the answer.
* The second index contains the contexts and their respective scores.

:::

We'll pass this query to the `answer_question` method and print the response.

```{python}
response = indox.answer_question(query=query,qa_model=mistral_qa)
```

**Answer:**

```{python}
response[0]
```

**Contexts and Scores:**

```{python}
response[1]
```

### Evaluation

Evaluating the performance of your question-answering system is crucial to ensure the quality and reliability of the responses. In this section, we will use the `Evaluation` module from Indox to assess our system's outputs.

```{python}
from Indox.Evaluation import Evaluation
evaluator = Evaluation(["BertScore", "Toxicity"])
```

#### Preparing Inputs for Evaluation

Next, we need to format the inputs according to the Indox evaluator's requirements. This involves creating a dictionary that includes the question, the generated answer, and the context from which the answer was derived.

```{python}
inputs = {
    "question" : query,
    "answer" : response[0],
    "context" : response[1][0]
}
result = evaluator(inputs)
```

#### Expected Output

```{python}
result
```

::: {.callout-tip} These metrics help you understand the accuracy and quality of the answers generated by your Indox RAG system. :::

***

Previous: Cluster Split Example
