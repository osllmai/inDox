[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDoxRag/blob/master/Demo/mistral_unstructured.ipynb)

# Mistral As Question Answer Model

## Introduction

In this notebook, we will demonstrate how to securely handle `inDoxRag` as
system for question answering system with open source models which are
available on internet like `Mistral`. so firstly you should buil
environment variables and API keys in Python using the `dotenv` library.
Environment variables are a crucial part of configuring your
applications, especially when dealing with sensitive information like
API keys.

Let\'s start by importing the required libraries and loading our
environment variables.

```python
!pip install mistralai
!pip install indoxRag
!pip install chromadb
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
```

### Import Essential Libraries

Then, we import essential libraries for our `IndoxRag` question answering
system:

- `IndoxRetrievalAugmentation`: Enhances the retrieval process for
  better QA performance.
- `Mistral`: A powerful QA model from IndoxRag, built on top of the
  semantic understanding.
- `UnstructuredLoadAndSplit`: A utility for loading and splitting
  unstructured data.

```python
from indoxRag import IndoxRetrievalAugmentation
indoxRag = IndoxRetrievalAugmentation()
```

:::

::: {#449eb2a7ca2e5bce .cell .markdown id="449eb2a7ca2e5bce"}

### Building the IndoxRag System and Initializing Models

Next, we will build our `inDoxRag` system and initialize the Mistral
question answering model along with the embedding model. This setup will
allow us to leverage the advanced capabilities of IndoxRag for our question
answering tasks.
:::

::: {#ac5ff6002e2497b3 .cell .code execution_count="4" ExecuteTime="{\"end_time\":\"2024-06-26T08:40:56.456996Z\",\"start_time\":\"2024-06-26T08:40:51.790145Z\"}" id="ac5ff6002e2497b3"}

```python
from indoxRag.llms import Mistral
from indoxRag.embeddings import MistralEmbedding
mistral_qa = Mistral(api_key=MISTRAL_API_KEY)
embed_mistral = MistralEmbedding(MISTRAL_API_KEY)
```

:::

::: {#fd23f48af26265ca .cell .markdown id="fd23f48af26265ca"}

### Setting Up Reference Directory and File Path

To demonstrate the capabilities of our IndoxRag question answering system,
we will use a sample directory. This directory will contain our
reference data, which we will use for testing and evaluation.

First, we specify the path to our sample file. In this case, we are
using a file named `sample.txt` located in our working directory. This
file will serve as our reference data for the subsequent steps.

Let\'s define the file path for our reference data.
:::

::: {#9706a7ba1cc8deff .cell .code}

```python
!wget https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
```

:::

::: {#b38c913b696a2642 .cell .code execution_count="5" ExecuteTime="{\"end_time\":\"2024-06-26T08:40:59.074020Z\",\"start_time\":\"2024-06-26T08:40:59.071498Z\"}" id="b38c913b696a2642"}

```python
file_path = "sample.txt"
```

:::

::: {#e88dd6c433fc600c .cell .markdown id="e88dd6c433fc600c"}

### Chunking Reference Data with UnstructuredLoadAndSplit

To effectively utilize our reference data, we need to process and chunk
it into manageable parts. This ensures that our question answering
system can efficiently handle and retrieve relevant information.

We use the `UnstructuredLoadAndSplit` utility for this task. This tool
allows us to load the unstructured data from our specified file and
split it into smaller chunks. This process enhances the performance of
our retrieval and QA models by making the data more accessible and
easier to process.

In this step, we define the file path for our reference data and use
`UnstructuredLoadAndSplit` to chunk the data with a maximum chunk size
of 400 characters.

Let\'s proceed with chunking our reference data.
:::

::: {#4dcc52c1d0416383 .cell .code execution_count="6" ExecuteTime="{\"end_time\":\"2024-06-26T08:41:58.334662Z\",\"start_time\":\"2024-06-26T08:41:58.008246Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4dcc52c1d0416383" outputId="c43a25f4-7c29-470c-8f82-6cfbb83be6d1"}

```python
from indoxRag.data_loader_splitter import UnstructuredLoadAndSplit
load_splitter = UnstructuredLoadAndSplit(file_path=file_path,max_chunk_size=400)
docs = load_splitter.load_and_chunk()
```

::: {.output .stream .stderr}
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data] Unzipping tokenizers/punkt.zip.
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data] /root/nltk_data...
[nltk_data] Unzipping taggers/averaged_perceptron_tagger.zip.
:::
:::

::: {#72d312cf4791f60f .cell .markdown id="72d312cf4791f60f"}

### Connecting Embedding Model to IndoxRag

With our reference data chunked and ready, the next step is to connect
our embedding model to the IndoxRag system. This connection enables the
system to leverage the embeddings for better semantic understanding and
retrieval performance.

We use the `connect_to_vectorstore` method to link the `embed_mistral`
model with our IndoxRag system. By specifying the embeddings and a
collection name, we ensure that our reference data is appropriately
indexed and stored, facilitating efficient retrieval during the
question-answering process.

Let\'s connect the embedding model to IndoxRag.
:::

::: {#ebc33cc4fb58a305 .cell .code execution_count="7" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:03.601445Z\",\"start_time\":\"2024-06-26T08:42:03.594047Z\"}" id="ebc33cc4fb58a305"}

```python
from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed_mistral)
```

:::

::: {#943f965096e65197 .cell .code execution_count="8" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:05.061630Z\",\"start_time\":\"2024-06-26T08:42:05.055595Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="943f965096e65197" outputId="0704ed8d-ace4-4112-c9dc-dbef9eab48b0"}

```python
indoxRag.connect_to_vectorstore(vectorstore_database=db)
```

::: {.output .execute_result execution_count="8"}
<indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x7fba6ca30280>
:::
:::

::: {#250da1a633bef038 .cell .markdown id="250da1a633bef038"}

### Storing Data in the Vector Store

After connecting our embedding model to the IndoxRag system, the next step
is to store our chunked reference data in the vector store. This process
ensures that our data is indexed and readily available for retrieval
during the question-answering process.

We use the `store_in_vectorstore` method to store the processed data in
the vector store. By doing this, we enhance the system\'s ability to
quickly access and retrieve relevant information based on the embeddings
generated earlier.

Let\'s proceed with storing the data in the vector store.
:::

::: {#83b2f51f1a359477 .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:15.228086Z\",\"start_time\":\"2024-06-26T08:42:07.249961Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="83b2f51f1a359477" outputId="c2d4c310-d550-4e15-9776-24b7c23a7ee8"}

```python
indoxRag.store_in_vectorstore(docs)
```

::: {.output .execute_result execution_count="9"}
<indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x7fba6ca30280>
:::
:::

::: {#7766ed35249fef6e .cell .markdown id="7766ed35249fef6e"}

## Query from RAG System with IndoxRag

With our Retrieval-Augmented Generation (RAG) system built using IndoxRag,
we are now ready to test it with a sample question. This test will
demonstrate how effectively our system can retrieve and generate
accurate answers based on the reference data stored in the vector store.

We\'ll use a sample query to test our system:

- **Query**: \"How did Cinderella reach her happy ending?\"

This question will be processed by our IndoxRag system to retrieve relevant
information and generate an appropriate response.

Let\'s test our RAG system with the sample question
:::

::: {#c30a41f4d7293b39 .cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:17.310350Z\",\"start_time\":\"2024-06-26T08:42:17.306685Z\"}" id="c30a41f4d7293b39"}

```python
query = "How cinderella reach her happy ending?"
```

:::

::: {#58639a3d46eb327f .cell .markdown id="58639a3d46eb327f"}
Now that our Retrieval-Augmented Generation (RAG) system with IndoxRag is
fully set up, we can test it with a sample question. We\'ll use the
`invoke` submethod to get a response from the system.

The `invoke` method processes the query using the connected QA model and
retrieves relevant information from the vector store. It returns a list
where:

- The first index contains the answer.
- The second index contains the contexts and their respective scores.

We\'ll pass this query to the `invoke` method and print the response.
:::

::: {#66ecb3768f04d326 .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:18.376680Z\",\"start_time\":\"2024-06-26T08:42:18.373295Z\"}" id="66ecb3768f04d326"}

```python
retriever = indoxRag.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5)
```

:::

::: {#9adbffdb7d5427bd .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:27.798069Z\",\"start_time\":\"2024-06-26T08:42:19.041579Z\"}" id="9adbffdb7d5427bd"}

```python
answer = retriever.invoke(query=query)
```

:::

::: {#f905f84906433aab .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-06-26T08:42:32.700407Z\",\"start_time\":\"2024-06-26T08:42:32.696411Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":140}" id="f905f84906433aab" outputId="da7808e4-2408-4ff6-b185-a97413f08713"}

```python
answer
```

::: {.output .execute_result execution_count="13"}

```json
{ "type": "string" }
```

:::
:::

::: {#db289e0dae276aee .cell .code execution_count="14" ExecuteTime="{\"end_time\":\"2024-06-09T10:23:16.751306Z\",\"start_time\":\"2024-06-09T10:23:16.746109Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="db289e0dae276aee" outputId="cafedfc6-1137-48ae-c095-b0052f393dcf"}

```python
context = retriever.context
context
```

::: {.output .execute_result execution_count="14"}
['by the hearth in the cinders. And as on that account she always\n\nlooked dusty and dirty, they called her cinderella.\n\nIt happened that the father was once going to the fair, and he\n\nasked his two step-daughters what he should bring back for them.\n\nBeautiful dresses, said one, pearls and jewels, said the second.\n\nAnd you, cinderella, said he, what will you have. Father',
"to appear among the number, they were delighted, called cinderella\n\nand said, comb our hair for us, brush our shoes and fasten our\n\nbuckles, for we are going to the wedding at the king's palace.\n\nCinderella obeyed, but wept, because she too would have liked to\n\ngo with them to the dance, and begged her step-mother to allow\n\nher to do so. You go, cinderella, said she, covered in dust and",
"danced with her only, and if any one invited her to dance, he said\n\nthis is my partner.\n\nWhen evening came, cinderella wished to leave, and the king's\n\nson was anxious to go with her, but she escaped from him so quickly\n\nthat he could not follow her. The king's son, however, had\n\nemployed a ruse, and had caused the whole staircase to be smeared",
'cinderella expressed a wish, the bird threw down to her what she\n\nhad wished for.\n\nIt happened, however, that the king gave orders for a festival\n\nwhich was to last three days, and to which all the beautiful young\n\ngirls in the country were invited, in order that his son might choose\n\nhimself a bride. When the two step-sisters heard that they too were',
"Then the maiden was delighted, and believed that she might now go\n\nwith them to the wedding. But the step-mother said, all this will\n\nnot help. You cannot go with us, for you have no clothes and can\n\nnot dance. We should be ashamed of you. On this she turned her\n\nback on cinderella, and hurried away with her two proud daughters.\n\nAs no one was now at home, cinderella went to her mother's"]
:::
:::

::: {#ea002e2ab9469d7b .cell .markdown id="ea002e2ab9469d7b"}

## Evaluation

Evaluating the performance of your question-answering system is crucial
to ensure the quality and reliability of the responses. In this section,
we will use the `Evaluation` module from IndoxRag to assess our system\'s
outputs.
:::

::: {#48d4718f798523c8 .cell .code ExecuteTime="{\"end_time\":\"2024-06-09T10:23:21.759373Z\",\"start_time\":\"2024-06-09T10:23:16.751306Z\"}" id="48d4718f798523c8"}

```python
from indoxRag.evaluation import Evaluation
evaluator = Evaluation(["BertScore", "Toxicity"])
```

:::

::: {#dca587b000e0f3d5 .cell .markdown id="dca587b000e0f3d5"}

### Preparing Inputs for Evaluation

Next, we need to format the inputs according to the IndoxRag evaluator\'s
requirements. This involves creating a dictionary that includes the
question, the generated answer, and the context from which the answer
was derived.
:::

::: {#26d130c534ed349f .cell .code ExecuteTime="{\"end_time\":\"2024-06-09T10:23:22.516004Z\",\"start_time\":\"2024-06-09T10:23:21.759373Z\"}" id="26d130c534ed349f"}

```python
inputs = {
    "question" : query,
    "answer" : answer,
    "context" : context
}
result = evaluator(inputs)
```

:::

::: {#da14c97311ae1028 .cell .code ExecuteTime="{\"end_time\":\"2024-06-09T10:23:22.534495Z\",\"start_time\":\"2024-06-09T10:23:22.516004Z\"}" id="da14c97311ae1028" outputId="e982f515-31c3-4b31-89c2-7351a34a67e2"}

```python
result
```

::: {.output .execute_result execution_count="15"}

```{=html}
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
```

:::
:::

::: {#1c0e58d968847693 .cell .code id="1c0e58d968847693"}

```python

```

:::
