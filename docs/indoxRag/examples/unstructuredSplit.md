# Unstructured Load And Split

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## Initial Setup

The following imports are essential for setting up the IndoxRag application. These imports include the main IndoxRag retrieval augmentation module, question-answering models, embeddings, and data loader splitter.

```python
from indoxRag import IndoxRetrievalAugmentation
indoxRag = IndoxRetrievalAugmentation()
```

### Generating response using OpenAI's language models

OpenAIQA class is used to handle question-answering task using OpenAI's language models. This instance creates OpenAiEmbedding class to specifying embedding model. Here ChromaVectorStore handles the storage and retrieval of vector embeddings by specifying a collection name and sets up a vector store where text embeddings can be stored and queried.

```python
from indoxRag.llms import OpenAiQA
from indoxRag.embeddings import OpenAiEmbedding
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
embed_openai = OpenAiEmbedding(api_key=OPENAI_API_KEY,model="text-embedding-3-small")

from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed_openai)
indoxRag.connect_to_vectorstore(vectorstore_database=db)
```

    <indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x1d0407cb440>

### load and preprocess data

This part of code demonstrates how to load and preprocess text data from a file, split it into chunks, and store these chunks in the vector store that was set up previously.

```python
file_path = "sample.txt"
```

```python
from indoxRag.data_loader_splitter import UnstructuredLoadAndSplit
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path,max_chunk_size=400)
docs = loader_splitter.load_and_chunk()
```

```python
indoxRag.store_in_vectorstore(docs=docs)
```

    <indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x1d0407cb440>

### Retrieve relevant information and generate an answer

The main purpose of these lines is to perform a query on the vector store to retrieve the most relevant information (top_k=5) and generate an answer using the language model.

```python
query = "How Cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=db, llm=openai_qa, top_k=5)
```

invoke(query) method sends the query to the retriever, which searches the vector store for relevant text chunks and uses the language model to generate a response based on the retrieved information.
Context property retrieves the context or the detailed information that the retriever used to generate the answer to the query. It provides insight into how the query was answered by showing the relevant text chunks and any additional information used.

```python
retriever.invoke(query)
retriever.context
```

    ["to appear among the number, they were delighted, called cinderella\n\nand said, comb our hair for us, brush our shoes and fasten our\n\nbuckles, for we are going to the wedding at the king's palace.\n\nCinderella obeyed, but wept, because she too would have liked to\n\ngo with them to the dance, and begged her step-mother to allow\n\nher to do so. You go, cinderella, said she, covered in dust and",
     "which they had wished for, and to cinderella he gave the branch\n\nfrom the hazel-bush. Cinderella thanked him, went to her mother's\n\ngrave and planted the branch on it, and wept so much that the tears\n\nfell down on it and watered it. And it grew and became a handsome\n\ntree. Thrice a day cinderella went and sat beneath it, and wept and\n\nprayed, and a little white bird always came on the tree, and if",
     'by the hearth in the cinders. And as on that account she always\n\nlooked dusty and dirty, they called her cinderella.\n\nIt happened that the father was once going to the fair, and he\n\nasked his two step-daughters what he should bring back for them.\n\nBeautiful dresses, said one, pearls and jewels, said the second.\n\nAnd you, cinderella, said he, what will you have. Father',
     "Then the maiden was delighted, and believed that she might now go\n\nwith them to the wedding. But the step-mother said, all this will\n\nnot help. You cannot go with us, for you have no clothes and can\n\nnot dance. We should be ashamed of you. On this she turned her\n\nback on cinderella, and hurried away with her two proud daughters.\n\nAs no one was now at home, cinderella went to her mother's",
     "danced with her only, and if any one invited her to dance, he said\n\nthis is my partner.\n\nWhen evening came, cinderella wished to leave, and the king's\n\nson was anxious to go with her, but she escaped from him so quickly\n\nthat he could not follow her. The king's son, however, had\n\nemployed a ruse, and had caused the whole staircase to be smeared"]

### With AgenticRag

AgenticRag stands for Agentic Retrieval-Augmented Generation. This concept combines retrieval-based methods and generation-based methods in natural language processing (NLP). The key idea is to enhance the generative capabilities of a language model by incorporating relevant information retrieved from a database or a vector store.
AgenticRag is designed to provide more contextually rich and accurate responses by utilizing external knowledge sources. It retrieves relevant pieces of information (chunks) from a vector store based on a query and then uses a language model to generate a comprehensive response that incorporates this retrieved information.

```python
agent = indoxRag.AgenticRag(llm=openai_qa,vector_database=db,top_k=5)
agent.run(query)
```

    Not Relevant doc
    Relevant doc
    Not Relevant doc
    Not Relevant doc
    Not Relevant doc





    "Cinderella reached her happy ending by receiving a branch from the hazel-bush from the prince, planting it on her mother's grave, and weeping and praying beneath it. The branch grew into a handsome tree, and a little white bird always came to the tree, bringing her comfort and assistance."

```python

```
