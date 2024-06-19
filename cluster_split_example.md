# Cluster\_Split\_Example

```{python}
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```

### Initial Setup

The following imports are essential for setting up the Indox application. These imports include the main Indox retrieval augmentation module, question-answering models, embeddings, and data loader splitter.

```{python}
from indox import IndoxRetrievalAugmentation
from indox.qa_models import OpenAiQA
from indox.embeddings import OpenAiEmbedding
from indox.data_loader_splitter import ClusteredSplit
```

In this step, we initialize the Indox Retrieval Augmentation, the QA model, and the embedding model. Note that the models used for QA and embedding can vary depending on the specific requirements.

```{python}
Indox = IndoxRetrievalAugmentation()
qa_model = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
embed = OpenAiEmbedding(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-small")
```

```{python}
file_path = "sample.txt"
```

### Data Loader Setup

We set up the data loader using the `ClusteredSplit` class. This step involves loading documents, configuring embeddings, and setting options for processing the text.

```{python}
docs = ClusteredSplit(file_path=file_path,embeddings=embed,remove_sword=True,re_chunk=False,chunk_size=300)
```

### Vector Store Connection and Document Storage

In this step, we connect the Indox application to the vector store and store the processed documents.

```{python}
Indox.connect_to_vectorstore(collection_name="sample",embeddings=embed)
```

```{python}
Indox.store_in_vectorstore(docs)
```

### Querying and Interpreting the Response

In this step, we query the Indox application with a specific question and use the QA model to get the response. The response is a tuple where the first element is the answer and the second element contains the retrieved context with their cosine scores. response\[0] contains the answer response\[1] contains the retrieved context with their cosine scores

```{python}
response = Indox.answer_question(query="How cinderella reach happy ending?",qa_model=qa_model,top_k=5)
```

```{python}
response[0]
```

```{python}
response[1]
```

***

Previous: Evaluation | Next: Mistral as Question Answering
